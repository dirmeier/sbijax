import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.abc.smcabc import smcabc


def test_smcabc_sample_returns_pytree():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Uniform(jnp.full(2, -3.0), jnp.full(2, 3.0))}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 0.1).sample(
      theta["theta"].shape, seed=seed
    )

  def summary(y):
    return y

  def distance(a, b):
    return jnp.linalg.norm(a - b, axis=-1)

  sampler = smcabc(prior, simulator, summary, distance)
  particles, info = sampler.sample(
    jr.PRNGKey(0), jnp.zeros((1, 2)), n_rounds=2, n_particles=100, ess_min=50
  )
  assert "theta" in particles
