"""Sequential neural posterior estimation.

A self-contained example of multi-round inference with the functional 0.4 API.
:func:`sbijax.run_sequential` simulates from the current posterior each round,
appends to the dataset, and refits; ``npe`` switches to its atomic
proposal-posterior loss in rounds > 0 automatically.
"""

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import npe, run_sequential
from sbijax.nn import make_maf


def prior_fn():
  return tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )


def simulator_fn(seed, theta):
  return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
    theta["theta"].shape, seed=seed
  )


def run():
  prior = prior_fn()
  estimator = npe(prior, make_maf(2))
  y_observed = jnp.array([-1.0, 1.0])

  params, info = run_sequential(
    jr.PRNGKey(0),
    estimator,
    prior,
    simulator_fn,
    y_observed,
    n_rounds=3,
    n_simulations_per_round=2_000,
  )
  print(f"finished round {info.round}")

  idata = estimator.sample(jr.PRNGKey(1), params, y_observed)
  theta = idata["/posterior"]["theta"].data.reshape(-1, 2)
  print("posterior mean:", jnp.mean(theta, axis=0))


if __name__ == "__main__":
  run()
