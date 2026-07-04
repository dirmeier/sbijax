"""Draw parameters and observations for simulation-based inference."""

import jax
from jax import numpy as jnp
from jax import random as jr


def simulate(rng_key, prior, simulator, *, proposal=None, n):
  """Draw a dataset of parameters and observations.

  Parameters are drawn from ``proposal`` if given, otherwise from ``prior``,
  and observations are produced by running ``simulator`` on them.

  Args:
      rng_key: a jax random key
      prior: a distribution to draw parameters from when ``proposal`` is None
      simulator: a callable ``(rng_key, theta) -> y``
      proposal: an optional callable ``(rng_key, n) -> theta`` used to draw
          parameters instead of the prior (e.g. a fitted posterior in a
          sequential round)
      n: the number of parameter/observation pairs to draw

  Returns:
      a dictionary with keys ``y`` and ``theta``
  """
  theta_key, sim_key = jr.split(rng_key)
  if proposal is None:
    theta = prior.sample(seed=theta_key, sample_shape=(n,))
  else:
    theta = proposal(theta_key, n)
  y = simulator(sim_key, theta)
  return {"y": y, "theta": theta}


def stack(data, new_data):
  """Append two datasets along the sample axis.

  Args:
      data: a dataset as returned by :func:`simulate`
      new_data: a second dataset with the same structure

  Returns:
      a dataset whose leaves are the concatenation of both inputs
  """
  return jax.tree_util.tree_map(
    lambda a, b: jnp.concatenate([a, b], axis=0), data, new_data
  )
