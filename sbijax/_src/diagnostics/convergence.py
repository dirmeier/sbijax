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
