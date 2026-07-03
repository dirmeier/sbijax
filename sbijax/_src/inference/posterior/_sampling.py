"""Shared posterior sampling for flow-based estimators."""

# ruff: noqa: PLR0913
import jax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.inference._sample_info import DirectSampleInfo


def rejection_sample_flow(
  rng_key, network, params, prior, observable, n_samples, *, n_sim=1024
):
  """Draw posterior samples from a flow, rejecting draws outside the prior.

  Repeatedly samples the conditional flow given ``observable`` and keeps only
  the proposals with finite prior density until ``n_samples`` are collected.

  Args:
      rng_key: a jax random key
      network: a flow with a ``sample`` method
      params: the fitted network parameters
      prior: the prior distribution used to reject invalid proposals
      observable: the observation to condition on
      n_samples: the number of accepted samples to return
      n_sim: the number of proposals drawn per rejection round

  Returns:
      a tuple ``(samples, DirectSampleInfo)`` of the named posterior pytree
      and a sampling record
  """
  observable = jnp.atleast_2d(observable)
  _, unravel_fn = ravel_pytree(prior.sample(seed=jr.PRNGKey(1)))
  thetas = None
  n_curr = n_samples
  while n_curr > 0:
    sample_key, rng_key = jr.split(rng_key)
    proposal = network.apply(
      params,
      sample_key,
      method="sample",
      context=jnp.tile(observable, [n_sim, 1]),
      is_training=False,
    )
    proposal_probs = prior.log_prob(jax.vmap(unravel_fn)(proposal))
    accepted = proposal[jnp.isfinite(proposal_probs)]
    thetas = accepted if thetas is None else jnp.vstack([thetas, accepted])
    n_curr -= accepted.shape[0]

  def reshape(p):
    if p.ndim == 1:
      p = p.reshape(p.shape[0], 1)
    return p.reshape(1, *p.shape)

  thetas = jax.tree_util.tree_map(
    reshape, jax.vmap(unravel_fn)(thetas[:n_samples])
  )
  return thetas, DirectSampleInfo(n_samples=n_samples)
