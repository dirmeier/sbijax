"""Shared posterior sampling for flow-based estimators."""

import jax
from jax import numpy as jnp

from sbijax._src.inference._sample_info import DirectSampleInfo


def rejection_sample_flow(rng_key, network, params, observable, n_samples):
  """Draw posterior samples directly from a conditional flow.

  Samples ``n_samples`` points from the flow conditioned on ``observable``
  in a single forward pass.

  Args:
      rng_key: a jax random key
      network: a flow with a ``sample`` method
      params: the fitted network parameters
      observable: the observation to condition on
      n_samples: the number of samples to draw

  Returns:
      a tuple ``(samples, DirectSampleInfo)`` where ``samples`` is a named
      posterior pytree of shape ``(1, n_samples, dim)`` and
      ``DirectSampleInfo`` is the sampling record
  """
  observable = jnp.atleast_2d(observable)
  thetas = network.apply(
    params,
    rng_key,
    method="sample",
    context=jnp.tile(observable, [n_samples, 1]),
    is_training=False,
  )

  def reshape(p):
    if p.ndim == 1:
      p = p.reshape(p.shape[0], 1)
    return p.reshape(1, *p.shape)

  thetas = jax.tree_util.tree_map(reshape, {"theta": thetas})
  return thetas, DirectSampleInfo(n_samples=n_samples)
