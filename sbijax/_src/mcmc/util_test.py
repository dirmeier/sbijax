# pylint: skip-file
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.mcmc.util import run_blackjax


class _FakeState(NamedTuple):
  position: dict


class _FakeInfo(NamedTuple):
  acceptance_rate: jax.Array


def _deterministic_init_fn(rng_key, initial_positions, lp):
  """A fake blackjax kernel: deterministically increments position by 1."""

  def kernel(rng_key, state):
    del rng_key
    new_position = jax.tree_util.tree_map(lambda x: x + 1.0, state.position)
    return _FakeState(new_position), _FakeInfo(jnp.array(1.0))

  return _FakeState(initial_positions), kernel


def test_run_blackjax_preserves_chain_identity():
  """Each chain's kept samples must be *that chain's* trajectory.

  Regression test for a bug where the post-warmup reshape used
  ``.reshape(n_chains, n_samples - n_warmup, -1)`` directly on an array
  shaped ``(n_samples, n_chains, dim)`` without swapping the leading two
  axes first, silently shuffling samples across chains and timesteps
  even though the output shape was unaffected.
  """
  n_chains, n_samples, n_warmup = 4, 20, 5
  # chains start far apart so a chain-identity mix-up is unmistakable.
  bases = jnp.array([0.0, 1_000.0, 2_000.0, 3_000.0])
  initial_positions = {"theta": bases[:, None]}

  thetas, _ = run_blackjax(
    jr.key(0),
    _deterministic_init_fn,
    initial_positions,
    lp=None,
    n_chains=n_chains,
    n_samples=n_samples,
    n_warmup=n_warmup,
  )

  # kernel increments by 1 each of `n_samples` steps, so absolute step t
  # (1-indexed) of chain c sits at `bases[c] + t`; post-warmup keeps
  # t = n_warmup + 1, ..., n_samples.
  expected = bases[:, None] + jnp.arange(n_warmup + 1, n_samples + 1)[None, :]
  assert jnp.allclose(thetas["theta"][:, :, 0], expected)
