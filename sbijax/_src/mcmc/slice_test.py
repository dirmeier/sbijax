import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference._sample_info import MCMCSampleInfo
from sbijax._src.mcmc.slice import sample_with_slice


def test_sample_with_slice_returns_samples_and_info():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def lp(theta):
    return jnp.sum(prior.log_prob(theta))

  samples, info = sample_with_slice(
    jr.PRNGKey(0), lp, prior, n_chains=2, n_samples=40, n_warmup=20
  )
  assert samples["theta"].shape == (2, 20, 2)
  assert isinstance(info, MCMCSampleInfo)
