import jax.numpy as jnp

from sbijax._src.inference._sample_info import DirectSampleInfo, MCMCSampleInfo


def test_mcmc_sample_info_fields():
  info = MCMCSampleInfo(acceptance_rate=jnp.array(0.8))
  assert abs(float(info.acceptance_rate) - 0.8) < 1e-6


def test_direct_sample_info_fields():
  info = DirectSampleInfo(n_samples=64)
  assert info.n_samples == 64
