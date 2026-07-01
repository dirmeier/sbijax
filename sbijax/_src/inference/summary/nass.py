"""Neural approximate sufficient statistics.

Implements the NASS method of :cite:t:`chen2023learning` as a functional
:class:`~sbijax._src.inference.summary._summary_net.SummaryNet`. The network
learns a summary of the data by maximising a Jensen-Shannon mutual-information
bound; the JSD loss is reused from the existing implementation.
"""

import jax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.inference.summary._summary_net import make_summary_net


def _jsd_summary_loss(params, rng, apply_fn, **batch):
  y, theta = batch["y"], batch["theta"]
  m, _ = y.shape
  summr = apply_fn(params, method="summary", y=y)
  idx_pos = jnp.tile(jnp.arange(m), 10)
  idx_neg = jax.vmap(lambda x: jr.permutation(x, m))(jr.split(rng, 10)).reshape(
    -1
  )
  f_pos = apply_fn(params, method="critic", y=summr, theta=theta)
  f_neg = apply_fn(
    params, method="critic", y=summr[idx_pos], theta=theta[idx_neg]
  )
  a, b = -jax.nn.softplus(-f_pos), jax.nn.softplus(f_neg)
  mi = a.mean() - b.mean()
  return -mi


def nass(network):
  """Construct a neural approximate sufficient statistics summary network.

  Args:
      network: a NASS summary network with ``forward``, ``summary`` and
          ``critic`` methods

  Returns:
      a :class:`~sbijax._src.inference.summary._summary_net.SummaryNet`
  """
  return make_summary_net(network, _jsd_summary_loss)
