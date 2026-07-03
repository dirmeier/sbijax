import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference._sample_info import MCMCSampleInfo
from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.simulate import simulate


def test_nle_sample_returns_pytree_and_mcmc_info():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  data = simulate(jr.PRNGKey(0), prior, simulator, n=200)
  est = nle(prior, make_maf(2))
  params, _ = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=100)
  samples, info = est.sample(
    jr.PRNGKey(2), params, jnp.zeros(2), n_chains=2, n_samples=40, n_warmup=20
  )
  assert samples["theta"].shape == (2, 20, 2)
  assert isinstance(info, MCMCSampleInfo)
