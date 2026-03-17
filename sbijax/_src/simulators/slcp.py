import chex
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from tensorflow_probability.substrates.jax import distributions as tfd


# ruff: noqa: PLR0913, E501
def slcp_model():
    """Simple likelihood complex posterior model.

    Constructs prior, simulator, and likelihood functions.

    Returns:
      returns a tuple of three objects. The first is a
      tfd.JointDistributionNamed serving as a prior distribution. The second
      is a simulator function that can be used to generate data. The third
      is the likelihood function.

    References:
      Papamakarios, George, Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows, 2019
    """

    def prior_fn():
        prior = tfd.JointDistributionNamed(
            dict(
                theta=tfd.Independent(
                    tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0)),
                    reinterpreted_batch_ndims=1,
                )
            )
        )
        return prior

    def likelihood(y, theta):
        mu = jnp.tile(theta[:2], 4)
        s1, s2 = theta[2] ** 2, theta[3] ** 2
        corr = s1 * s2 * jnp.tanh(theta[4])
        cov = jnp.array([[s1**2, corr], [corr, s2**2]])
        cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
        lik_fn = tfd.MultivariateNormalFullCovariance(mu, cov)
        log_lik = lik_fn.log_prob(y)
        return log_lik

    def simulator(seed, theta):
        theta = theta["theta"]
        chex.assert_rank(theta, 2)
        us_key, noise_key = jr.split(seed)

        def _unpack_params(ps):
            m0 = ps[..., [0]]
            m1 = ps[..., [1]]
            s0 = ps[..., [2]] ** 2
            s1 = ps[..., [3]] ** 2
            r = np.tanh(ps[..., [4]])
            return m0, m1, s0, s1, r

        m0, m1, s0, s1, r = _unpack_params(theta)
        us = tfd.Normal(0.0, 1.0).sample(
            seed=us_key, sample_shape=(theta.shape[0], 4, 2)
        )
        xs = jnp.empty_like(us)
        xs = xs.at[..., 0].set(s0 * us[..., 0] + m0)
        y = xs.at[..., 1].set(
            s1 * (r * us[..., 0] + np.sqrt(10 - r**2) * us[..., 1]) + m1
        )
        y = y.reshape((*theta.shape[:1], 8))
        return y

    return prior_fn(), simulator, likelihood
