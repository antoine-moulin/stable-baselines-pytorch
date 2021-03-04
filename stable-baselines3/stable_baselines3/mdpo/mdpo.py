import warnings
from typing import Any, Dict, Optional, Type, Union, Tuple

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, get_policy_from_name
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from mdpo_utils import get_distribution, log_q


class MDPO(OnPolicyAlgorithm):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(MDPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

        #         policy_base = None
        #         if isinstance(policy, str) and policy_base is not None:
        #             self.policy_class = get_policy_from_name(policy_base, policy)
        #         else:
        #             self.policy_class = policy
        self.policy_class = ActorCriticPolicy

        self.old_policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs
        )

    def _setup_model(self) -> None:
        super(MDPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []

        # # Construct network for new policy
        # self.policy_pi = self.policy_class(self.observation_space, self.action_space, self.n_envs, 1,
        #                              None, reuse=False, **self.policy_kwargs)
        #
        # # Network for old policy
        # with tf.variable_scope("oldpi", reuse=False):
        #     self.old_policy = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
        #                                   None, reuse=False, **self.policy_kwargs)
        #
        #
        # with tf.variable_scope("loss", reuse=False):
        #     self.atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        #     self.vtarg = tf.placeholder(dtype=tf.float32, shape=[None])
        #     self.ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
        #     self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
        #     self.outer_learning_rate_ph = tf.placeholder(tf.float32, [], name="outer_learning_rate_ph")
        #     self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
        #     self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")
        #
        #     observation = self.policy_pi.obs_ph
        #     self.action = self.policy_pi.pdtype.sample_placeholder([None])
        #
        #     if self.tsallis_q == 1.0:
        #         # kloldnew = self.old_policy.proba_distribution.kl(self.policy.proba_distribution)
        #         kloldnew = self.policy_pi.proba_distribution.kl(self.old_policy.proba_distribution)
        #         ent = self.policy_pi.proba_distribution.entropy()
        #         meankl = tf.reduce_mean(kloldnew)
        #
        #     else:
        #         logp_pi = self.policy_pi.proba_distribution.logp(self.action)
        #         logp_pi_old = self.old_policy.proba_distribution.logp(self.action)
        #         ent = self.policy_pi.proba_distribution.entropy()
        #         # kloldnew = self.policy_pi.proba_distribution.kl_tsallis(self.old_policy.proba_distribution, self.tsallis_q)
        #         tsallis_q = 2.0 - self.tsallis_q
        #         meankl = tf.reduce_mean(tf_log_q(tf.exp(logp_pi), tsallis_q) - tf_log_q(tf.exp(logp_pi_old),
        #                                                                                 tsallis_q))  # tf.reduce_mean(kloldnew)
        #
        #     meanent = tf.reduce_mean(ent)
        #     entbonus = self.entcoeff * meanent
        #
        #     if self.cliprange_vf is None:
        #         vpred_clipped = self.policy_pi.value_flat
        #     else:
        #         vpred_clipped = self.old_vpred_ph + \
        #                         tf.clip_by_value(self.policy_pi.value_flat - self.old_vpred_ph,
        #                                          - self.clip_range_vf_ph, self.clip_range_vf_ph)
        #
        #     vf_losses1 = tf.square(self.policy_pi.value_flat - self.ret)
        #     vf_losses2 = tf.square(vpred_clipped - self.ret)
        #     vferr = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        #
        #     # advantage * pnew / pold
        #     ratio = tf.exp(self.policy_pi.proba_distribution.logp(self.action) -
        #                    self.old_policy.proba_distribution.logp(self.action))
        #
        #     if self.method == "multistep-SGD":
        #         surrgain = tf.reduce_mean(ratio * self.atarg) - meankl / self.learning_rate_ph
        #     elif self.method == "closedreverse-KL":
        #         surrgain = tf.reduce_mean(tf.exp(self.atarg) * self.policy_pi.proba_distribution.logp(self.action))
        #
        #     optimgain = surrgain  # + entbonus - self.learning_rate_ph * meankl
        #     losses = [optimgain, meankl, entbonus, surrgain, meanent]
        #     self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]
        #
        #     dist = meankl
        #
        #     all_var_list = tf_util.get_trainable_vars("model")
        #     var_list = [v for v in all_var_list if "/vf" not in v.name and "/q/" not in v.name]
        #     vf_var_list = [v for v in all_var_list if "/pi" not in v.name and "/logstd" not in v.name]
        #     print("policy vars", var_list)
        #
        #     all_closed_var_list = tf_util.get_trainable_vars("closedpi")
        #     closed_var_list = [v for v in all_closed_var_list if "/vf" not in v.name and "/q" not in v.name]
        #
        #     self.get_flat = tf_util.GetFlat(var_list, sess=self.sess)
        #     self.set_from_flat = tf_util.SetFromFlat(var_list, sess=self.sess)
        #
        #     klgrads = tf.gradients(dist, var_list)
        #     flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        #     shapes = [var.get_shape().as_list() for var in var_list]
        #     start = 0
        #     tangents = []
        #     for shape in shapes:
        #         var_size = tf_util.intprod(shape)
        #         tangents.append(tf.reshape(flat_tangent[start: start + var_size], shape))
        #         start += var_size
        #     gvp = tf.add_n([tf.reduce_sum(grad * tangent)
        #                     for (grad, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
        #     fvp = tf_util.flatgrad(gvp, var_list)
        #
        #     tf.summary.scalar('entropy_loss', meanent)
        #     tf.summary.scalar('policy_gradient_loss', optimgain)
        #     tf.summary.scalar('value_function_loss', surrgain)
        #     tf.summary.scalar('approximate_kullback-leibler', meankl)
        #     tf.summary.scalar('loss', optimgain + meankl + entbonus + surrgain + meanent)
        #
        #     self.assign_old_eq_new = \
        #         tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
        #                                           zipsame(tf_util.get_globals_vars("oldpi"),
        #                                                   tf_util.get_globals_vars("model"))])
        #     self.compute_losses = tf_util.function(
        #         [observation, self.old_policy.obs_ph, self.action, self.atarg, self.learning_rate_ph, self.vtarg],
        #         losses)
        #     self.compute_fvp = tf_util.function(
        #         [flat_tangent, observation, self.old_policy.obs_ph, self.action, self.atarg],
        #         fvp)
        #     self.compute_vflossandgrad = tf_util.function(
        #         [observation, self.old_policy.obs_ph, self.ret, self.old_vpred_ph, self.clip_range_vf_ph],
        #         tf_util.flatgrad(vferr, vf_var_list))
        #
        #     grads = tf.gradients(-optimgain, var_list)
        #     grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
        #     trainer = tf.train.AdamOptimizer(learning_rate=self.outer_learning_rate_ph, epsilon=1e-5)
        #     # trainer = tf.train.AdamOptimizer(learning_rate=3e-4, epsilon=1e-5)
        #     grads = list(zip(grads, var_list))
        #     self._train = trainer.apply_gradients(grads)
        #
        #     @contextmanager
        #     def timed(msg):
        #         if self.rank == 0 and self.verbose >= 1:
        #             print(colorize(msg, color='magenta'))
        #             start_time = time.time()
        #             yield
        #             print(colorize("done in {:.3f} seconds".format((time.time() - start_time)),
        #                            color='magenta'))
        #         else:
        #             yield
        #
        #     def allmean(arr):
        #         assert isinstance(arr, np.ndarray)
        #         out = np.empty_like(arr)
        #         MPI.COMM_WORLD.Allreduce(arr, out, op=MPI.SUM)
        #         out /= self.nworkers
        #         return out
        #
        #     tf_util.initialize(sess=self.sess)
        #
        #     th_init = self.get_flat()
        #     MPI.COMM_WORLD.Bcast(th_init, root=0)
        #     self.set_from_flat(th_init)
        #
        # with tf.variable_scope("Adam_mpi", reuse=False):
        #     self.vfadam = MpiAdam(vf_var_list, sess=self.sess)
        #     if self.using_gail:
        #         self.d_adam = MpiAdam(self.reward_giver.get_trainable_variables(), sess=self.sess)
        #         self.d_adam.sync()
        #     self.vfadam.sync()
        #
        # with tf.variable_scope("input_info", reuse=False):
        #     tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.ret))
        #     tf.summary.scalar('learning_rate', tf.reduce_mean(self.vf_stepsize))
        #     tf.summary.scalar('advantage', tf.reduce_mean(self.atarg))
        #     tf.summary.scalar('kl_clip_range', tf.reduce_mean(self.max_kl))
        #
        #     if self.full_tensorboard_log:
        #         tf.summary.histogram('discounted_rewards', self.ret)
        #         tf.summary.histogram('learning_rate', self.vf_stepsize)
        #         tf.summary.histogram('advantage', self.atarg)
        #         tf.summary.histogram('kl_clip_range', self.max_kl)
        #         if tf_util.is_image(self.observation_space):
        #             tf.summary.image('observation', observation)
        #         else:
        #             tf.summary.histogram('observation', observation)
        #
        # self.timed = timed
        # self.allmean = allmean
        #
        # self.step = self.policy_pi.step
        # self.proba_step = self.policy_pi.proba_step
        # self.initial_state = self.policy_pi.initial_state
        #
        # self.params = tf_util.get_trainable_vars("model") + tf_util.get_trainable_vars("oldpi")
        # if self.using_gail:
        #     self.params.extend(self.reward_giver.get_trainable_variables())
        #
        # self.summary = tf.summary.merge_all()
        #
        # self.compute_lossandgrad = \
        #     tf_util.function(
        #         [observation, self.old_policy.obs_ph, self.action, self.atarg, self.ret, self.learning_rate_ph,
        #          self.vtarg, self.closed_policy.obs_ph],
        #         [self.summary, tf_util.flatgrad(optimgain, var_list)] + losses)

        self.tsallis_q = 1.0  # TODO

        rollout_data = list(self.rollout_buffer.get(self.n_steps * self.n_envs))[0]

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        if self.tsallis_q == 1.0:
            distribution_pi = get_distribution(self.policy, rollout_data.observations)
            distribution_old_pi = get_distribution(self.old_policy, rollout_data.observations)

            kloldnew = th.distributions.kl.kl_divergence(distribution_pi.distribution, distribution_old_pi.distribution)
            ent = distribution_pi.entropy()
            meankl = kloldnew.mean()

            log_pi = distribution_pi.log_prob(actions)
            logp_pi_old = rollout_data.old_log_prob

        else:
            _, log_pi, ent = self.policy.evaluate_actions(rollout_data.observations, actions)
            logp_pi_old = rollout_data.old_log_prob
            # ent = self.policy_pi.proba_distribution.entropy()
            # kloldnew = self.policy_pi.proba_distribution.kl_tsallis(self.old_policy.proba_distribution, self.tsallis_q)
            tsallis_q = 2.0 - self.tsallis_q
            meankl = th.mean(log_q(th.exp(logp_pi), tsallis_q) - log_q(th.exp(logp_pi_old), tsallis_q))

        meanent = th.mean(ent)
        entbonus = self.ent_coef * meanent

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values, - self.clip_range_vf, self.clip_range_vf
            )

        value_loss1 = F.mse_loss(rollout_data.returns, values, reduction="none")
        value_loss2 = F.mse_loss(rollout_data.returns, values_pred, reduction="none")
        vferr = th.mean(th.maximum(value_loss1, value_loss2))

        value_losses.append(vferr.item())

        # advantage * pnew / pold
        ratio = th.exp(log_pi - log_pi_old)

        advantages = rollout_data.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.method == "multistep-SGD":
            surrgain = th.mean(ratio * advantages) - meankl / self.learning_rate_ph
        elif self.method == "closedreverse-KL":
            surrgain = tf.reduce_mean(tf.exp(self.atarg) * self.policy_pi.proba_distribution.logp(self.action))
        else:
            policygain = tf.reduce_mean(tf.exp(self.atarg) * tf.log(self.closed_policy.proba_distribution.mean))
            surrgain = tf.reduce_mean(ratio * self.atarg) - tf.reduce_mean(
                self.learning_rate_ph * ratio * self.policy_pi.proba_distribution.logp(self.action))

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss = - th.min(advantages * ratio).mean()

                # Logging
                pg_losses.append(policy_loss.item())

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        # update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MDPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "MDPO":

        return super(MDPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

