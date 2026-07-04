"""Neural approximate slice sufficient statistics.

Implements the NASSS method of :cite:t:`chen2021neural` as a functional
``SummaryFns``. It differs from NASS only in the
loss (a slice-based JSD bound with a secondary summary); the training and
summarization logic are shared.
"""

import jax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.inference.summary._summary_net import make_summary_net


def _sample_unit_sphere(rng_key, n, dim):
  u = jr.normal(rng_key, (n, dim))
  norm = jnp.linalg.norm(u, ord=2, axis=-1, keepdims=True)
  return u / norm


def _jsd_summary_loss(params, rng_key, apply_fn, **batch):
  y, theta = batch["y"], batch["theta"]
  n, p = theta.shape
  phi_key, rng_key = jr.split(rng_key)
  summr = apply_fn(params, method="summary", y=y)
  summr = jnp.tile(summr, [10, 1])
  theta = jnp.tile(theta, [10, 1])
  phi = _sample_unit_sphere(phi_key, 10, p)
  phi = jnp.repeat(phi, n, axis=0)
  second_summr = apply_fn(
    params, method="secondary_summary", y=summr, theta=phi
  )
  theta_prime = jnp.sum(theta * phi, axis=1).reshape(-1, 1)
  idx_pos = jnp.tile(jnp.arange(n), 10)
  perm_key, rng_key = jr.split(rng_key)
  idx_neg = jax.vmap(lambda x: jr.permutation(x, n))(
    jr.split(perm_key, 10)
  ).reshape(-1)
  f_pos = apply_fn(params, method="critic", y=second_summr, theta=theta_prime)
  f_neg = apply_fn(
    params,
    method="critic",
    y=second_summr[idx_pos],
    theta=theta_prime[idx_neg],
  )
  a, b = -jax.nn.softplus(-f_pos), jax.nn.softplus(f_neg)
  mi = a.mean() - b.mean()
  return -mi


def nasss(network):
  """Construct a neural approximate slice sufficient statistics summary network.

  Args:
      network: a NASSS summary network with ``forward``, ``summary``,
          ``secondary_summary`` and ``critic`` methods

  Returns:
      a ``SummaryFns``
  """
  return make_summary_net(network, _jsd_summary_loss)
