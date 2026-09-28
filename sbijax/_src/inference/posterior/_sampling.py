"""Shared posterior sampling for flow-based estimators."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax.flatten_util import ravel_pytree

from sbijax._src.inference._sample_info import DirectSampleInfo


def reject_outside_support(rng_key, draw_fn, n_samples, prior=None):
  """Draw flat posterior samples that lie inside the prior's support.

  Draws batches of ``n_samples`` points from ``draw_fn`` and keeps those with
  a finite prior log-density until ``n_samples`` are kept. Without a prior
  every draw is kept.

  Args:
      rng_key: a jax random key
      draw_fn: a callable ``(rng_key, n) -> draws`` returning ``n`` flat
          parameter vectors
      n_samples: the number of samples to return
      prior: the prior whose support the samples must lie in, or ``None``

  Returns:
      a tuple ``(samples, DirectSampleInfo)`` where ``samples`` is a named
      posterior pytree of shape ``(1, n_samples, dim)`` and
      ``DirectSampleInfo`` is the sampling record

  Raises:
      ValueError: if a batch has no draw inside the prior's support
  """
  if prior is not None:
    _, unravel_fn = ravel_pytree(prior.sample(seed=jr.key(0)))

  kept, n_kept, n_drawn = [], 0, 0
  while n_kept < n_samples:
    draw_key, rng_key = jr.split(rng_key)
    thetas = draw_fn(draw_key, n_samples).reshape(n_samples, -1)
    n_drawn += n_samples
    if prior is not None:
      lp = prior.log_prob(jax.vmap(unravel_fn)(thetas))
      thetas = thetas[jnp.isfinite(lp)]
      if thetas.shape[0] == 0:
        raise ValueError(
          f"none of {n_samples} posterior draws lies inside the prior's support"
        )
    kept.append(thetas)
    n_kept += thetas.shape[0]

  thetas = jnp.concatenate(kept, axis=0)[:n_samples]
  return {"theta": thetas[None]}, DirectSampleInfo(
    n_samples=n_samples, acceptance_rate=n_kept / n_drawn
  )


# ruff: noqa: PLR0913
def rejection_sample_flow(
  rng_key, network, params, observable, n_samples, prior=None
):
  """Draw posterior samples from a conditional flow by rejection.

  Samples points from the flow conditioned on ``observable`` and rejects those
  outside the prior's support.

  Args:
      rng_key: a jax random key
      network: a flow with a ``sample`` method
      params: the fitted network parameters
      observable: the observation to condition on
      n_samples: the number of samples to draw
      prior: the prior whose support the samples must lie in, or ``None``

  Returns:
      a tuple ``(samples, DirectSampleInfo)`` where ``samples`` is a named
      posterior pytree of shape ``(1, n_samples, dim)`` and
      ``DirectSampleInfo`` is the sampling record
  """
  observable = jnp.atleast_2d(observable)

  def draw_fn(rng_key, n):
    return network.apply(
      params,
      rng_key,
      method="sample",
      context=jnp.tile(observable, [n, 1]),
      is_training=False,
    )

  return reject_outside_support(rng_key, draw_fn, n_samples, prior)
