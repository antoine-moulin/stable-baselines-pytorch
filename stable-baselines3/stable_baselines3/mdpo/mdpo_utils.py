import pytorch as th


def get_distribution(policy, obs):

    latent_pi, latent_vf, latent_sde = policy._get_latent(obs)
    distribution = policy._get_action_dist_from_latent(latent_pi, latent_sde)

    return distribution

def log_q(x,q):
    safe_x = th.maximum(x, 1e-6)

    log_q_x = th.where(th.equal(q, 1.), th.log(safe_x), (th.pow(safe_x, 1 - q) - 1) / (1 - q))

    return log_q_x