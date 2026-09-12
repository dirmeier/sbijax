import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.util.data import flatten_chains, unravel_draws


def _multi_leaf_prior():
  return tfd.JointDistributionNamed(
    {
      "variance": tfd.InverseGamma(
        concentration=3.0 * jnp.ones(1), scale=2.0 * jnp.ones(1)
      ),
      "mean": lambda variance: tfd.Normal(jnp.zeros(2), jnp.sqrt(variance)),
    },
    batch_ndims=0,
  )


def test_flatten_chains_collapses_chain_and_draw_axes():
  samples = {"theta": jnp.ones((3, 5, 2))}
  flat = flatten_chains(samples)
  assert flat["theta"].shape == (15, 2)


def test_unravel_draws_splits_flat_draws_by_sorted_key():
  flat = jnp.arange(24.0).reshape(1, 8, 3)
  named = unravel_draws({"theta": flat}, _multi_leaf_prior())

  assert set(named) == {"mean", "variance"}
  assert named["mean"].shape == (1, 8, 2)
  assert named["variance"].shape == (1, 8, 1)
  # ravel_pytree orders leaves by sorted key, so mean takes columns 0-1
  assert jnp.allclose(named["mean"], flat[..., :2])
  assert jnp.allclose(named["variance"], flat[..., 2:])


def test_unravel_draws_is_identity_on_matching_structure():
  # what the mcmc and abc methods already return
  draws = {"mean": jnp.zeros((2, 5, 2)), "variance": jnp.ones((2, 5, 1))}
  assert unravel_draws(draws, _multi_leaf_prior()) is draws


def test_unravel_draws_is_identity_for_a_single_theta_prior():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )
  draws = {"theta": jnp.zeros((1, 4, 2))}
  assert unravel_draws(draws, prior) is draws
