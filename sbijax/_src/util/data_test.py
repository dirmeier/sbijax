import jax.numpy as jnp

from sbijax._src.util.data import flatten_chains


def test_flatten_chains_collapses_chain_and_draw_axes():
  samples = {"theta": jnp.ones((3, 5, 2))}
  flat = flatten_chains(samples)
  assert flat["theta"].shape == (15, 2)
