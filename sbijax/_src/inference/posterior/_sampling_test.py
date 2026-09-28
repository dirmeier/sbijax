import jax.numpy as jnp
import pytest
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.posterior._sampling import reject_outside_support


def _unit_square_prior():
  return tfd.JointDistributionNamed(
    {"theta": tfd.Uniform(jnp.zeros(2), jnp.ones(2))}, batch_ndims=0
  )


def _standard_normal_draws(rng_key, n):
  return jr.normal(rng_key, (n, 2))


def test_reject_outside_support_keeps_only_draws_in_support():
  samples, info = reject_outside_support(
    jr.key(0), _standard_normal_draws, 1_000, _unit_square_prior()
  )
  theta = samples["theta"]
  assert theta.shape == (1, 1_000, 2)
  assert jnp.all((theta >= 0.0) & (theta <= 1.0))
  # a standard normal falls in [0, 1] with probability 0.3413 per dimension
  assert abs(info.acceptance_rate - 0.3413**2) < 0.02


def test_reject_outside_support_keeps_every_draw_without_prior():
  samples, info = reject_outside_support(
    jr.key(0), _standard_normal_draws, 1_000
  )
  assert samples["theta"].shape == (1, 1_000, 2)
  assert info.acceptance_rate == 1.0


def test_reject_outside_support_raises_without_draws_in_support():
  with pytest.raises(ValueError, match="prior's support"):
    reject_outside_support(
      jr.key(0),
      lambda rng_key, n: jnp.full((n, 2), -1.0),
      100,
      _unit_square_prior(),
    )
