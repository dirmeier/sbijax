"""Convergence diagnostics re-exported from blackjax (DR-015).

Both functions operate on the named samples pytree returned by
``Estimator.sample`` for MCMC methods, whose leaves have shape
``(n_chains, n_draws, dim)``.
"""

import jax
from blackjax.diagnostics import (
  effective_sample_size,
  potential_scale_reduction,
)


def rhat(samples):
  """Split-R-hat for each parameter dimension.

  Args:
      samples: a named pytree with leaves of shape ``(n_chains, n_draws, dim)``

  Returns:
      a pytree of the same structure with per-dimension R-hat values
  """
  return jax.tree_util.tree_map(
    lambda x: potential_scale_reduction(x, chain_axis=0, sample_axis=1),
    samples,
  )


def ess(samples):
  """Effective sample size for each parameter dimension.

  Args:
      samples: a named pytree with leaves of shape ``(n_chains, n_draws, dim)``

  Returns:
      a pytree of the same structure with per-dimension ESS values
  """
  return jax.tree_util.tree_map(
    lambda x: effective_sample_size(x, chain_axis=0, sample_axis=1),
    samples,
  )


def mcmc_convergence(samples, n_chains):
  """Convergence diagnostics for a set of MCMC draws, guarded on chain count.

  R-hat is a between-chain statistic and is undefined for a single chain; ESS
  is guarded together with it so both diagnostics travel as a pair.

  Args:
      samples: a named pytree with leaves of shape ``(n_chains, n_draws, dim)``
      n_chains: the number of chains the draws were sampled from

  Returns:
      a tuple ``(rhat, ess)`` of per-dimension pytrees when ``n_chains > 1``,
      otherwise ``(None, None)``
  """
  if n_chains < 2:
    return None, None
  return rhat(samples), ess(samples)
