"""Neural posterior score estimation example.

Demonstrates NPSE on a simple mixture model.
"""

import argparse

import optax
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import npse, sample, simulate, train
from sbijax.experimental.nn import make_score_model


def prior_fn():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), jnp.array(1.0))}, batch_ndims=0
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


def run(n_iter):
  prior = prior_fn()
  y_observed = jnp.array([-2.0, 1.0])
  neural_network = make_score_model(2)
  estimator = npse(neural_network)

  data = simulate(jr.key(1), prior, simulator_fn, n=10_000)
  params, info = train(
    jr.key(2),
    estimator,
    data,
    optimizer=optax.adam(3e-4),
    n_early_stopping_patience=25,
    n_iter=n_iter,
  )
  samples, _ = sample(jr.key(3), estimator, params, y_observed)
  theta = samples["theta"].reshape(-1, samples["theta"].shape[-1])
  print("posterior mean:", jnp.mean(theta, axis=0))
  print("posterior std: ", jnp.std(theta, axis=0))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-iter", type=int, default=10)
  args = parser.parse_args()
  run(args.n_iter)
