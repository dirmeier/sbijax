"""Posterior samplers bundling an MCMC kernel, a prior, and Gaussian init."""

from collections.abc import Callable
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.mcmc.util import run_blackjax


class Kernel(NamedTuple):
  """A BlackJAX kernel handle.

  Attributes:
      init_fn: ``(rng_key, initial_positions, lp) -> (initial_states, kernel)``
  """

  init_fn: Callable


def _gaussian_init(prior, rng_key, n_chains):
  """``N(0, 1)`` initial positions with the prior's pytree structure."""
  template = prior.sample(seed=jr.key(0))
  leaves, treedef = jax.tree_util.tree_flatten(template)
  keys = jr.split(rng_key, len(leaves))
  drawn = [
    jr.normal(k, (n_chains, *jnp.shape(leaf)))
    for k, leaf in zip(keys, leaves, strict=True)
  ]
  return jax.tree_util.tree_unflatten(treedef, drawn)


def make_sampler(kernel, *, prior, **kernel_kwargs):
  """Build a posterior sampler from an MCMC ``kernel`` and a ``prior``.

  Args:
      kernel: a :class:`Kernel` handle (e.g. ``sbijax.mcmc.nuts``)
      prior: the prior; used for the target density and Gaussian chain init
      **kernel_kwargs: forwarded to the kernel

  Returns:
      a callable ``(rng_key, loglik_fn, *, n_chains, n_samples, n_warmup) ->
      (samples, MCMCSampleInfo)`` where the target is
      ``loglik_fn(theta) + prior.log_prob(theta)`` and chains start at
      ``N(0, I)``.
  """

  def sampler(
    rng_key, loglik_fn, *, n_chains=4, n_samples=2_000, n_warmup=1_000
  ):
    def logdensity(theta):
      return jnp.sum(loglik_fn(theta)) + jnp.sum(prior.log_prob(theta))

    init_key, sample_key = jr.split(rng_key)
    positions = _gaussian_init(prior, init_key, n_chains)
    return run_blackjax(
      sample_key,
      kernel.init_fn,
      positions,
      logdensity,
      n_chains=n_chains,
      n_samples=n_samples,
      n_warmup=n_warmup,
    )

  return sampler
