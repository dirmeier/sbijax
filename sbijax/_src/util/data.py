import jax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten

from sbijax._src.util.types import PyTree


def _tree_stack(trees):
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = tree_flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)

  grouped_leaves = zip(*leaves_list, strict=False)
  result_leaves = [jnp.vstack(leave) for leave in grouped_leaves]
  return jax.tree_util.tree_unflatten(treedef_list[0], result_leaves)


def stack_data(data: PyTree, also_data: PyTree) -> PyTree:
  """Stack two data sets.

  Args:
      data: one data set
      also_data: another data set

  Returns:
      returns the stack of the two data sets
  """
  if data is None:
    return also_data
  if also_data is None:
    return data
  stacked = _tree_stack([data, also_data])
  return stacked


def flatten_chains(samples: PyTree) -> PyTree:
  """Collapse the ``(n_chains, n_draws, dim)`` sample axes into ``(N, dim)``.

  Args:
      samples: a named pytree of posterior draws with a leading chain and draw
          axis on every leaf

  Returns:
      the same pytree with each leaf reshaped to ``(n_chains * n_draws, dim)``
  """
  return jax.tree_util.tree_map(lambda x: x.reshape(-1, x.shape[-1]), samples)


def unravel_draws(samples: PyTree, prior) -> PyTree:
  """Reshape flat posterior draws into the prior's named pytree.

  The amortized estimators emit the flattened parameter vector under the
  single key ``"theta"``, because they are built from a network and never see
  the prior. MCMC and ABC methods emit the prior's named pytree. This maps the
  former onto the latter, using the ``ravel_pytree`` leaf order the networks
  were trained on.

  Args:
      samples: a pytree of draws, each leaf shaped ``(n_chains, n_draws, dim)``
      prior: the prior whose structure the draws should take

  Returns:
      the draws in the prior's structure; ``samples`` unchanged if it is
      already in that structure, which covers MCMC, ABC and any prior that is
      itself a single ``"theta"`` leaf
  """
  probe = prior.sample(seed=jr.key(0))
  if jax.tree_util.tree_structure(samples) == jax.tree_util.tree_structure(
    probe
  ):
    return samples
  _, unravel_fn = ravel_pytree(probe)
  return jax.vmap(jax.vmap(unravel_fn))(samples["theta"])
