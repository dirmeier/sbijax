"""Posterior samplers bundling an MCMC kernel, a prior, and chain init."""

from collections.abc import Callable
from typing import NamedTuple

from jax import numpy as jnp
from jax import random as jr

from sbijax._src.mcmc.util import run_blackjax


class Kernel(NamedTuple):
  """Identifies a BlackJAX MCMC algorithm (NUTS, MALA, RMH, IMH).

  Wraps the algorithm's initializer so :func:`make_sampler` can build the
  concrete kernel and initial chain states at sampling time. The single field
  ``init_fn`` has signature
  ``(rng_key, initial_positions, lp) -> (initial_states, kernel)``.
  """

  init_fn: Callable


def _prior_init(prior, rng_key, n_chains):
  """Initial chain positions drawn from the prior.

  Drawing from the prior rather than from ``N(0, 1)`` keeps every leaf inside
  its own support: a constrained parameter (e.g. a ``HalfNormal`` scale) would
  otherwise be initialised at a negative value, making the target density
  ``-inf`` there, which collapses the adapted step size to zero and freezes
  that chain at its starting point.
  """
  return prior.sample(seed=rng_key, sample_shape=(n_chains,))


def make_sampler(kernel, *, prior, **kernel_kwargs):
  """Build a posterior sampler from an MCMC ``kernel`` and a ``prior``.

  Args:
      kernel: a ``Kernel`` handle (e.g. ``sbijax.mcmc.nuts``)
      prior: the prior; used for the target density and chain init
      **kernel_kwargs: forwarded to the kernel

  Returns:
      a callable ``(rng_key, loglik_fn, *, n_chains, n_samples, n_warmup) ->
      (samples, MCMCSampleInfo)`` where the target is
      ``loglik_fn(theta) + prior.log_prob(theta)`` and chains start at draws
      from the prior.
  """

  def sampler(
    rng_key, loglik_fn, *, n_chains=4, n_samples=2_000, n_warmup=1_000
  ):
    def logdensity(theta):
      return jnp.sum(loglik_fn(theta)) + jnp.sum(prior.log_prob(theta))

    init_key, sample_key = jr.split(rng_key)
    positions = _prior_init(prior, init_key, n_chains)
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
