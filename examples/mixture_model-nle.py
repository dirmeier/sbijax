"""Neural likelihood estimation example.

Demonstrates NLE on a simple mixture model.
"""

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import nle, simulate
from sbijax.nn import make_mdn, make_spf


def prior_fn():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1)}, batch_ndims=0
  )
  return prior


def simulator_fn(seed, theta):
  mean = theta["theta"].reshape(-1, 2)
  n = mean.shape[0]
  data_key, cat_key = jr.split(seed)
  categories = tfd.Categorical(logits=jnp.zeros(2)).sample(
    seed=cat_key, sample_shape=(n,)
  )
  scales = jnp.array([1.0, 0.1])[categories].reshape(-1, 1)
  y = tfd.Normal(mean, scales).sample(seed=data_key)
  return y


def run(use_spf, n_iter):
  prior = prior_fn()
  y_observed = jnp.array([-2.0, 1.0])
  neural_network = (
    make_spf(2, -5.0, 5.0, n_params=10) if use_spf else make_mdn(2, 10)
  )
  model = nle(prior, neural_network)

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
  parser.add_argument("--use-spf", action="store_true", default=False)
  parser.add_argument("--n-iter", type=int, default=1_000)
  args = parser.parse_args()
  run(args.use_spf, args.n_iter)
