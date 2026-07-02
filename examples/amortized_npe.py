"""Amortized neural posterior estimation.

A self-contained example of the functional 0.4 API: define a prior and a
simulator, draw a training set, fit an ``npe`` estimator, and sample the
posterior for an observation.
"""

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import npe, simulate
from sbijax.nn import make_maf


def prior_fn():
  return tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )


def simulator_fn(seed, theta):
  return theta["theta"] + tfd.Normal(0.0, 0.1).sample(
    theta["theta"].shape, seed=seed
  )


def run():
  prior = prior_fn()
  estimator = npe(prior, make_maf(2))

  data = simulate(jr.PRNGKey(0), prior, simulator_fn, n=10_000)
  params, info = estimator.fit(jr.PRNGKey(1), data)
  print(f"trained for {info.losses.shape[0]} epochs")

  y_observed = jnp.array([-1.0, 1.0])
  idata = estimator.sample(jr.PRNGKey(2), params, y_observed)
  theta = idata["/posterior"]["theta"].data.reshape(-1, 2)
  print("posterior mean:", jnp.mean(theta, axis=0))


if __name__ == "__main__":
  run()
