from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import blackjax
import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.mcmc.sampler import Kernel
from sbijax._src.mcmc.util import burn_in, run_blackjax

if TYPE_CHECKING:
  from collections.abc import Callable

  from tensorflow_probability.substrates.jax import distributions as tfd

  from sbijax._src.inference._sample_info import MCMCSampleInfo


class _SliceStepInfo(NamedTuple):
  acceptance_rate: Array


# ruff: noqa: PLR0913, D417
def sample_with_slice(
  rng_key: Array,
  lp: Callable[[Any], Array],
  prior: tfd.Distribution,
  *,
  n_chains: int = 4,
  n_samples: int = 2_000,
  n_warmup: int = 1_000,
  **kwargs: Any,
) -> tuple[Any, MCMCSampleInfo]:
  r"""Sample from a distribution using a slice sampler.

  Args:
      rng_key: a jax random key
      lp: the logdensity you wish to sample from
      prior: a function that returns a prior sample
      n_chains: number of chains to sample
      n_samples: number of samples per chain returned after the warmup
      n_warmup: number of samples to discard
      **kwargs: the kernel options ``n_thin`` (number of extra slice moves
          discarded between draws, default 2), ``n_doubling`` (maximum number
          of interval doublings, default 5) and ``step_size`` (initial
          interval width, default 1.0)

  Examples:
      >>> import functools as ft
      >>> from jax import numpy as jnp, random as jr
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      ...
      >>> prior = tfd.JointDistributionNamed(
      ...     dict(
      ...         mean=tfd.Normal(jnp.zeros(2), 1.0),
      ...         std=tfd.HalfNormal(1.0)
      ...     ),
      ...     batch_ndims=0,
      ... )
      >>> def log_prob(theta, y):
      ...     lp_prior = prior.log_prob(theta)
      ...     lp_data = tfd.Normal(theta["mean"], theta["std"]).log_prob(y)
      ...     return jnp.sum(lp_data) + jnp.sum(lp_prior)
      ...
      >>> prop_posterior_lp = ft.partial(log_prob, y=jnp.array([-1.0, 1.0]))
      >>> samples = sample_with_slice(jr.PRNGKey(0), prop_posterior_lp, prior)

  Returns:
      a tuple ``(samples, info)``: a named pytree with leaves of shape
      ``n_chains x n_samples x dim`` and an ``MCMCSampleInfo``
      whose ``acceptance_rate`` is the fraction of slice moves that found a
      point within the shrinkage budget (close to one for well-behaved
      targets)
  """
  init_key, run_key = jr.split(rng_key)
  initial_positions = prior.sample(seed=init_key, sample_shape=(n_chains,))
  return run_blackjax(
    run_key,
    _slice_init,
    initial_positions,
    lp,
    n_chains=n_chains,
    n_samples=n_samples,
    n_warmup=n_warmup,
    **kwargs,
  )


def _slice_init(
  rng_key: Array,
  initial_positions: Any,
  lp: Callable[[Any], Array],
  n_warmup: int,
  *,
  n_thin: int = 2,
  n_doubling: int = 5,
  step_size: float = 1.0,
) -> tuple[Any, Callable[[Array, Any], tuple[Any, _SliceStepInfo]]]:
  """Initialise and warm up slice sampling chains.

  Args:
      rng_key: a jax random key
      initial_positions: a named pytree of chain start positions
      lp: the logdensity to sample from
      n_warmup: number of single slice moves discarded as warmup
      n_thin: number of extra slice moves discarded between draws
      n_doubling: maximum number of interval doublings
      step_size: initial width of the slice interval

  Returns:
      the chain states after warmup and a ``(keys, states) -> (states, info)``
      function that advances every chain by one thinned draw

  Raises:
      ValueError: if ``n_thin`` is negative or ``step_size`` is not positive
  """
  if n_thin < 0:
    raise ValueError(f"n_thin must be non-negative, got {n_thin}")
  if step_size <= 0:
    raise ValueError(f"step_size must be positive, got {step_size}")
  algorithm = blackjax.slice_sampling(
    lp, width=step_size, max_expansions=n_doubling
  )
  single = jax.vmap(algorithm.step)
  states = jax.vmap(algorithm.init)(initial_positions)
  states = burn_in(rng_key, single, states, n_warmup)

  def thinned(keys, states):
    def body(states, t):
      sub_keys = jax.vmap(jr.fold_in, in_axes=(0, None))(keys, t)
      states, info = single(sub_keys, states)
      return states, info.is_accepted

    states, accepted = jax.lax.scan(body, states, jnp.arange(n_thin + 1))
    return states, _SliceStepInfo(
      acceptance_rate=jnp.mean(accepted.astype(jnp.float32), axis=0)
    )

  return states, thinned


slice_sampling = Kernel(init_fn=_slice_init)
