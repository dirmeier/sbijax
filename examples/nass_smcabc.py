"""NASS+SMCABC example.

Demonstrates neural approximate sufficient statistics with SMCABC on the
simple likelihood complex posterior model.
"""

import argparse

import jax
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import nass, simulate, smcabc, train
from sbijax.nn import make_nass_net


def prior_fn():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))}, batch_ndims=0
  )
  return prior


def simulator_fn(seed, theta):
  theta = theta["theta"]
  orig_shape = theta.shape
  if theta.ndim == 2:
    theta = theta[:, None, :]
  us_key, noise_key = jr.split(seed)

  def _unpack_params(ps):
    m0 = ps[..., [0]]
    m1 = ps[..., [1]]
    s0 = ps[..., [2]] ** 2
    s1 = ps[..., [3]] ** 2
    r = jnp.tanh(ps[..., [4]])
    return m0, m1, s0, s1, r

  m0, m1, s0, s1, r = _unpack_params(theta)
  us = tfd.Normal(0.0, 1.0).sample(
    seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
  )
  xs = jnp.empty_like(us)
  xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
  y = xs.at[:, :, :, 1].set(
    s1 * (r * us[:, :, :, 0] + jnp.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
  )
  if len(orig_shape) == 2:
    y = y.reshape((*theta.shape[:1], 8))
  else:
    y = y.reshape((*theta.shape[:2], 8))
  return y


def distance_fn(y_simulated, y_observed):
  diff = y_simulated - y_observed
  dist = jax.vmap(jnp.linalg.norm)(diff)
  return dist


def run(n_rounds, n_iter):
  prior = prior_fn()
  y_observed = jnp.array(
    [
      [
        -0.9707123,
        -2.9461224,
        -0.4494722,
        -3.4231849,
        -0.13285634,
        -3.364017,
        -0.85367596,
        -2.4271638,
      ]
    ]
  )

  summary_net = nass(make_nass_net(5, (64, 64)))
  data = simulate(jr.key(1), prior, simulator_fn, n=20_000)
  params_nass, _ = train(
    jr.key(2), summary_net, data, n_early_stopping_patience=25, n_iter=n_iter
  )

  def summary_fn(y):
    return summary_net.summarize_fn(params_nass, y)

  smc = smcabc(prior, simulator_fn, summary_fn, distance_fn)
  particles, _ = smc.sample(
    jr.key(3),
    y_observed,
    n_rounds=n_rounds,
    n_particles=5_000,
    eps_step=0.825,
    ess_min=2_000,
  )
  theta = particles["theta"].reshape(-1, particles["theta"].shape[-1])
  print("posterior mean:", jnp.mean(theta, axis=0))
  print("posterior std: ", jnp.std(theta, axis=0))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-rounds", type=int, default=15)
  parser.add_argument("--n-iter", type=int, default=1_000)
  args = parser.parse_args()
  run(args.n_rounds, args.n_iter)
