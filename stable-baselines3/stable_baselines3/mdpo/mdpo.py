import time
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

from stable_baselines3.mdpo.mdpo_utils import get_distribution, log_q
import copy


class MDPO(OnPolicyAlgorithm):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
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
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

        #         policy_base = None
        #         if isinstance(policy, str) and policy_base is not None:
        #             self.policy_class = get_policy_from_name(policy_base, policy)
        #         else:
        #             self.policy_class = policy
        self.policy_class = ActorCriticPolicy  # TODO: turn into attribute

        self.old_policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs
        )

        # update old policy
        self.old_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

        self.tsallis_q = 1.0  # TODO: turn into attribute
        self.method = 'multistep-SGD'  # TODO: turn into attribute

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

        t_start = time.time()

        # for param in self.old_policy.parameters():
        #     param.requires_grad = False

        #### Policy Optimization -------------------------------------------------------------------
        # TODO: the optimization below should be done `sgd_steps` times, and not just once
        # For the policy network, the whole buffer is used in an update
        rollout_data = list(self.rollout_buffer.get(self.n_steps * self.n_envs))[0]

        # Actions
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        # Compute the log probabilities and the values
        values, log_pi, ent = self.policy.evaluate_actions(rollout_data.observations, actions)

        # Old log probabilities
        logp_pi_old = rollout_data.old_log_prob

        # Compute the KL divergence between the current policy and the old policy
        if self.tsallis_q == 1.0:
            distribution_pi = get_distribution(self.policy, rollout_data.observations)
            distribution_old_pi = get_distribution(self.old_policy, rollout_data.observations)

            kloldnew = th.distributions.kl.kl_divergence(distribution_pi.distribution, distribution_old_pi.distribution)
            meankl = kloldnew.mean()

        # Compute the Tsallis divergence between the current policy and the old policy
        else:
            tsallis_q = 2.0 - self.tsallis_q
            meankl = th.mean(log_q(th.exp(log_pi), tsallis_q) - log_q(th.exp(logp_pi_old), tsallis_q))

        # Compute entropy bonus
        # TODO: try to incorporate it in the loss later
        meanent = th.mean(ent)
        entbonus = self.ent_coef * meanent

        # Compute the ratio of the policies and the advantages to compute: advantages * ratios
        ratio = th.exp(log_pi - logp_pi_old)

        advantages = rollout_data.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute the (penalized) loss function
        if self.method == "multistep-SGD":
            # Anneal the step size from 1 to 0
            lr_now = self._current_progress_remaining + self.batch_size / self.total_timesteps

            # TODO: check what to do with lr_now (replace with inverse?)
            surrgain = th.mean(ratio * advantages) - meankl / lr_now

        # Compute ...
        elif self.method == "closedreverse-KL":
            surrgain = th.mean(th.exp(advantages) * log_pi)

        # Compute ...
        else:
            # policygain = th.mean(th.exp(advantages) * th.log(self.policy.mean_actions))
            surrgain = th.mean(ratio * advantages) - th.mean(
                self.learning_rate_ph * ratio * log_pi)

        # TODO: check this...
        for name, param in self.policy.named_parameters():
            # Optimize over the policy's parameters only
            # print(name)
            if "policy" not in name and "action" not in name:
                param.requires_grad = False

        with th.no_grad():
            policy_tmp = copy.deepcopy(self.policy.state_dict())

        # print("#########"+str(self._n_updates)+"#######")
        # print(list(self.policy.parameters()))
        # Should be optimizing over the policy's parameters only
        self.policy.optimizer.zero_grad()
        surrgain.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        # print("###### After update #####")
        # print(list(self.policy.parameters()))
        #### Value Optimization -------------------------------------------------------------------
        # TODO: check this...
        for name, param in self.policy.named_parameters():
            # Optimize over the value's parameters only
            if name is not None:
                if "value" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        vflosses = []  # to log the losses

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()
            # Compute the values
            values, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)

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

            # # TODO: check this...
            # for name, param in self.policy.named_parameters():
            #     # Optimize over the value's parameters only
            #     if name is not None:
            #         if "value" not in name:
            #             param.requires_grad = False
            #         else:
            #             param.requires_grad = True

            # Should be optimizing over the value's parameters only
            self.policy.optimizer.zero_grad()
            vferr.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            vflosses.append(vferr)  # logging

        value_losses.append(th.mean(th.Tensor(vflosses)).item())  # logging


        # TODO: where to update the old policy without causing an error?
        # update old policy
        # with th.no_grad():
        self.old_policy.load_state_dict(policy_tmp)
        #
        # for param in self.old_policy.parameters():
        #     param.requires_grad = False


        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        # logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", surrgain.item())
        logger.record("train/value_loss", value_losses[-1])
        logger.record("train/mean_kl", meankl.item())
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

        self.total_timesteps = total_timesteps

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
