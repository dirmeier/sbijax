from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd


# ruff: noqa: PLR0913, E501
def mixture_model_with_distractors():
    """Mixture model with distractors.

    Constructs prior, simulator, and likelihood functions.

    Returns:
      returns a tuple of three objects. The first is a
      tfd.JointDistributionNamed serving as a prior distribution. The second
      is a simulator function that can be used to generate data. The third
      is the likelihood function.

    References:
      Albert, Carlo, et al., Simulated Annealing ABC with multiple summary statistics, 2025
    """
    alpha = 0.3
    sigma = 0.3

    def prior_fn():
        return tfd.JointDistributionNamed(
            dict(theta=tfd.Uniform(jnp.array([-10.0]), jnp.array([10.0])))
        )

    def simulator(seed, theta):
        parameters = theta["theta"].reshape(-1, 1)
        neg_parameters = -theta["theta"].reshape(-1, 1)
        idxs_rng_key, seed = jr.split(seed)
        idxs = (
            tfd.Categorical(probs=jnp.array([alpha, 1.0 - alpha]))
            .sample(
                seed=idxs_rng_key,
                sample_shape=(
                    parameters.shape[0],
                    2,
                ),
            )
            .reshape(-1, 2)
        )
        means = jnp.concatenate((parameters, neg_parameters), axis=1)
        means = jnp.take_along_axis(means, idxs, axis=1)
        means = means.squeeze()
        scales = jnp.array([1.0, sigma])[idxs.squeeze()]
        distr = tfd.Normal(loc=means, scale=scales)

        y_rng_key, distrators_rng_key, seed = jr.split(seed, 3)
        y = distr.sample(seed=y_rng_key).reshape(-1, 2)
        distractor = (
            tfd.Normal(0.0, 1.0)
            .sample(seed=distrators_rng_key, sample_shape=(y.shape[0], 8))
            .reshape(-1, 8)
        )
        y = jnp.concatenate((y, distractor), axis=1)
        return y

    def likelihood(y, theta):
        y = y.reshape(-1, 10)[:, :2]
        theta = theta["theta"].reshape(-1, 1)
        theta = jnp.broadcast_to(theta, y.shape)
        lp1 = tfd.Normal(loc=theta, scale=1.0).log_prob(y)
        lp2 = tfd.Normal(loc=-theta, scale=sigma).log_prob(y)
        lp = jnp.logaddexp(jnp.log(alpha) + lp1, jnp.log(1 - alpha) + lp2)
        lp = lp.sum(axis=1)
        return lp

    return prior_fn(), simulator, likelihood
