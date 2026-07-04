# pylint: skip-file

import chex
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.simulate.simulate import simulate, stack


def _prior():
  return tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )


def _simulator(seed, theta):
  return theta["theta"] + tfd.Normal(0.0, 0.1).sample(
    theta["theta"].shape, seed=seed
  )


def test_simulate_from_prior_shapes():
  data = simulate(jr.PRNGKey(0), _prior(), _simulator, n=128)
  chex.assert_shape(data["y"], (128, 2))
  chex.assert_shape(data["theta"]["theta"], (128, 2))


def test_simulate_from_proposal_uses_proposal_not_prior():
  def proposal(rng_key, n):
    return {"theta": jnp.zeros((n, 2))}

  data = simulate(jr.PRNGKey(0), _prior(), _simulator, proposal=proposal, n=64)
  chex.assert_shape(data["theta"]["theta"], (64, 2))
  chex.assert_trees_all_close(data["theta"]["theta"], jnp.zeros((64, 2)))


def test_stack_appends_rounds():
  a = simulate(jr.PRNGKey(0), _prior(), _simulator, n=10)
  b = simulate(jr.PRNGKey(1), _prior(), _simulator, n=6)
  both = stack(a, b)
  chex.assert_shape(both["y"], (16, 2))
  chex.assert_shape(both["theta"]["theta"], (16, 2))
