"""Sequential Monte Carlo ABC example.

Demonstrates sequential Monte Carlo ABC on a simple bivariate Gaussian example.
"""

import argparse

import jax
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import smcabc


def prior_fn():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), jnp.ones(2))}, batch_ndims=0
  )
  return prior


def simulator_fn(seed, theta):
  p = tfd.Normal(jnp.zeros_like(theta["theta"]), 0.1)
  y = theta["theta"] + p.sample(seed=seed)
  return y


def summary_fn(y):
  return y


def distance_fn(y_simulated, y_observed):
  diff = y_simulated - y_observed
  dist = jax.vmap(jnp.linalg.norm)(diff)
  return dist


def run(n_rounds):
  prior = prior_fn()
  y_observed = jnp.array([-1.0, 1.0])

  smc = smcabc(prior, simulator_fn, summary_fn, distance_fn)
  particles, _ = smc.sample(
    jr.key(1),
    y_observed,
    n_rounds=1,
    n_particles=1000,
    ess_min=500,
    eps_step=0.9,
  )
  theta = particles["theta"].reshape(-1, particles["theta"].shape[-1])
  print("posterior mean:", jnp.mean(theta, axis=0))
  print("posterior std: ", jnp.std(theta, axis=0))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-rounds", type=int, default=10)
  args = parser.parse_args()
  run(args.n_rounds)
