import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference._sample_info import MCMCSampleInfo
from sbijax._src.mcmc.nuts import nuts
from sbijax._src.mcmc.sampler import make_sampler


def test_make_sampler_draws_from_target_with_gaussian_init():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )
  sampler = make_sampler(nuts, prior=prior)

  def loglik(theta):
    return (
      tfd.Normal(jnp.array([1.0, -1.0]), 1.0).log_prob(theta["theta"]).sum()
    )

  samples, info = sampler(
    jr.key(0), loglik, n_chains=4, n_samples=200, n_warmup=100
  )
  assert samples["theta"].shape == (4, 100, 2)
  assert isinstance(info, MCMCSampleInfo)
