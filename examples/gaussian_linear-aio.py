"""All-in-one simulation-based inference.

Demonstrates AiO on a linear Gaussian model.
"""

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import simulate
from sbijax.experimental import aio
from sbijax.experimental.nn import make_simformer_based_score_model

import numpy as np


def prior_fn():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(5), 1)}, batch_ndims=0
  )
  return prior


def simulator_fn(seed, theta):
  mean = theta["theta"].reshape(-1, 5)
  y = tfd.Normal(mean, 0.1).sample(seed=seed)
  return y


def run(n_iter):
  prior = prior_fn()
  y_observed = jnp.linspace(-2.0, 2.0, 5)
  mask = jnp.zeros((10, 10))
  mask = mask.at[np.arange(5, 10), np.arange(5)].set(1)
  mask = mask + mask.T + jnp.eye(10)

  neural_network = make_simformer_based_score_model(5, mask, 1, 1)
  model = aio(prior, neural_network)

  data = simulate(jr.PRNGKey(1), prior, simulator_fn, n=10_000)
  params, info = model.fit(
    jr.PRNGKey(2), data, n_early_stopping_patience=25, n_iter=n_iter
  )
  samples, _ = model.sample(jr.PRNGKey(3), params, y_observed)
  theta = samples["theta"].reshape(-1, samples["theta"].shape[-1])
  print("posterior mean:", jnp.mean(theta, axis=0))
  print("posterior std: ", jnp.std(theta, axis=0))


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--n-iter", type=int, default=1_000)
  args = parser.parse_args()
  run(args.n_iter)
