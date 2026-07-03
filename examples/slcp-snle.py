"""Surjective neural likelihood estimation example.

Demonstrates sequential surjective neural likelihood estimation on the simple
 likelihood complex posterior model.
"""

import argparse

import haiku as hk
import jax
import optax
import surjectors
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from surjectors import (
  AffineMaskedAutoregressiveInferenceFunnel,
  Chain,
  MaskedAutoregressive,
  Permutation,
  TransformedDistribution,
)
from surjectors.nn import MADE, make_mlp
from surjectors.util import unstack
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import run_sequential, snle


def prior_fn():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))}, batch_ndims=0
  )
  return prior


def simulator_fn(seed, theta):
  theta = theta["theta"]
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
  y = y.reshape((*theta.shape[:1], 8))
  return y


def likelihood_fn(theta, y):
  mu = jnp.tile(theta[:2], 4)
  s1, s2 = theta[2] ** 2, theta[3] ** 2
  corr = s1 * s2 * jnp.tanh(theta[4])
  cov = jnp.array([[s1**2, corr], [corr, s2**2]])
  cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
  p = tfd.MultivariateNormalFullCovariance(mu, cov)
  return p.log_prob(y)


def log_density_fn(theta, y):
  prior_lp = tfd.Independent(
    tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0)), 1
  ).log_prob(theta)
  likelihood_lp = likelihood_fn(theta, y)

  lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
  return lp


def make_model(dim, use_surjectors):
  def _bijector_fn(params):
    means, log_scales = unstack(params, -1)
    return surjectors.ScalarAffine(means, jnp.exp(log_scales))

  def _decoder_fn(n_dim):
    decoder_net = make_mlp(
      [50, n_dim * 2],
      w_init=hk.initializers.TruncatedNormal(stddev=0.001),
    )

    def _fn(z):
      params = decoder_net(z)
      mu, log_scale = jnp.split(params, 2, -1)
      return tfd.Independent(tfd.Normal(mu, jnp.exp(log_scale)), 1)

    return _fn

  def _flow(method, **kwargs):
    layers = []
    n_dimension = dim
    order = jnp.arange(n_dimension)
    for i in range(5):
      if i == 2 and use_surjectors:
        n_latent = 6
        layer = AffineMaskedAutoregressiveInferenceFunnel(
          n_latent,
          _decoder_fn(n_dimension - n_latent),
          conditioner=MADE(
            n_latent,
            [64, 64],
            2,
            w_init=hk.initializers.TruncatedNormal(0.001),
            b_init=jnp.zeros,
            activation=jax.nn.tanh,
          ),
        )
        n_dimension = n_latent
        order = order[::-1]
        order = order[:n_dimension] - jnp.min(order[:n_dimension])
      else:
        layer = MaskedAutoregressive(
          bijector_fn=_bijector_fn,
          conditioner=MADE(
            n_dimension,
            [64, 64],
            2,
            w_init=hk.initializers.TruncatedNormal(0.001),
            b_init=jnp.zeros,
            activation=jax.nn.tanh,
          ),
        )
        order = order[::-1]
      layers.append(layer)
      layers.append(Permutation(order, 1))
    chain = Chain(layers)

    base_distribution = tfd.Independent(
      tfd.Normal(jnp.zeros(n_dimension), jnp.ones(n_dimension)),
      reinterpreted_batch_ndims=1,
    )
    td = TransformedDistribution(base_distribution, chain)
    return td(method, **kwargs)

  td = hk.transform(_flow)
  td = hk.without_apply_rng(td)
  return td


def run(n_rounds, n_iter):
  prior = prior_fn()
  y_obs = jnp.array(
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

  neural_network = make_model(8, use_surjectors=True)
  estimator = snle(prior, neural_network)
  optimizer = optax.adam(1e-3)

  params, info = run_sequential(
    jr.PRNGKey(1),
    estimator,
    prior,
    simulator_fn,
    y_obs,
    n_rounds=n_rounds,
    n_simulations_per_round=2_000,
    optimizer=optimizer,
    n_iter=n_iter,
  )

  samples, _ = estimator.sample(jr.PRNGKey(3), params, y_obs)
  theta = samples["theta"].reshape(-1, samples["theta"].shape[-1])
  print("posterior mean:", jnp.mean(theta, axis=0))
  print("posterior std: ", jnp.std(theta, axis=0))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-rounds", type=int, default=15)
  parser.add_argument("--n-iter", type=int, default=1_000)
  args = parser.parse_args()
  run(args.n_rounds, args.n_iter)
