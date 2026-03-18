from functools import partial

import jax
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.python.internal.backend.jax.gen.linear_operator_lower_triangular import (
  LinearOperatorLowerTriangular,
)
from tensorflow_probability.substrates.jax import distributions as tfd

m11 = jnp.array([-0.5, 0.0])
m12 = jnp.array([0.5, 0.0])
m21 = jnp.array([0.0, -0.5])
m22 = jnp.array([0.0, 0.5])
scale = jnp.array(0.1)
nu = 3.0


def _eudclidean(theta, m1, m2):
  diff = jnp.linalg.norm(theta - m1, ord=2) - jnp.linalg.norm(theta - m2, ord=2)
  return jnp.repeat(jnp.abs(diff), 10)


dists_1_fn = jax.vmap(partial(_eudclidean, m1=m11, m2=m12))
dists_2_fn = jax.vmap(partial(_eudclidean, m1=m21, m2=m22))


# ruff: noqa: PLR0913, E501
def hyperboloid():
  """Hyperboloid model.

  Constructs prior, simulator, and likelihood functions.

  Returns:
    returns a tuple of three objects. The first is a
    tfd.JointDistributionNamed serving as a prior distribution. The second
    is a simulator function that can be used to generate data. The third
    is the likelihood function.

  References:
    Forbes, Florence, et al., Summary statistics and discrepancy measures for approximate Bayesian computation via surrogate posteriors, 2022
  """

  def prior_fn():
    return tfd.JointDistributionNamed(
      dict(
        theta=tfd.Independent(
          tfd.Uniform(jnp.full(2, -2.0), jnp.full(2, 2.0)), 1
        )
      )
    )

  def simulator(seed, theta):
    mix_key, data_key = jr.split(seed)
    theta = theta["theta"].reshape(-1, 2)
    d1 = dists_1_fn(theta).reshape(-1, 1, 10)
    d2 = dists_2_fn(theta).reshape(-1, 1, 10)
    theta = jnp.concatenate([d1, d2], axis=1)
    idxs = jr.categorical(mix_key, logits=jnp.ones(2), shape=(theta.shape[0],))
    idxs = idxs.reshape(-1, 1, 1)
    locs = jnp.take_along_axis(theta, idxs, 1).squeeze()
    scales = scale * jnp.eye(10)
    distr = tfd.MultivariateStudentTLinearOperator(
      df=nu,
      loc=locs,
      scale=LinearOperatorLowerTriangular(scales),
    )
    y = distr.sample(seed=data_key)
    return y.reshape(-1, 10)

  def likelihood(y, theta):
    theta = theta["theta"].reshape(-1, 2)
    d1 = dists_1_fn(theta).reshape(-1, 10)
    d2 = dists_2_fn(theta).reshape(-1, 10)
    scales = scale * jnp.eye(10)
    lp1 = tfd.MultivariateStudentTLinearOperator(
      df=nu,
      loc=d1.squeeze(),
      scale=LinearOperatorLowerTriangular(scales),
    ).log_prob(y)
    lp2 = tfd.MultivariateStudentTLinearOperator(
      df=nu,
      loc=d2.squeeze(),
      scale=LinearOperatorLowerTriangular(scales),
    ).log_prob(y)
    lp = jnp.logaddexp(jnp.log(0.5) + lp1, jnp.log(0.5) + lp2)
    return lp

  return prior_fn(), simulator, likelihood
