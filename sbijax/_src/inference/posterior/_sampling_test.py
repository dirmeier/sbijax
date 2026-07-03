import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference._sample_info import DirectSampleInfo
from sbijax._src.inference.posterior.fmpe import fmpe
from sbijax._src.nn.make_continuous_flow import make_cnf
from sbijax._src.simulate import simulate


def test_fmpe_sample_returns_pytree_and_direct_info():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  data = simulate(jr.PRNGKey(0), prior, simulator, n=200)
  est = fmpe(prior, make_cnf(2))
  params, _ = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=100)
  samples, info = est.sample(jr.PRNGKey(2), params, jnp.zeros(2), n_samples=64)
  assert samples["theta"].shape == (1, 64, 2)
  assert isinstance(info, DirectSampleInfo) and info.n_samples == 64
