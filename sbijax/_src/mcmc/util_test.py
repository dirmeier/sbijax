# pylint: skip-file

import blackjax as bj
import chex
import jax
import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference._sample_info import MCMCSampleInfo
from sbijax._src.mcmc.nuts import sample_with_nuts
from sbijax._src.mcmc.util import run_blackjax


def _mala_init(rng_key, n_chains, prior, lp):
  initial_positions = prior.sample(seed=rng_key, sample_shape=(n_chains,))
  kernel = bj.mala(lp, 0.1)
  initial_state = jax.vmap(kernel.init)(initial_positions)
  return initial_state, kernel.step


def test_run_blackjax_returns_chain_shaped_samples(prior_log_prob_tuple):
  prior_fn, lp = prior_log_prob_tuple
  samples, info = run_blackjax(
    jr.PRNGKey(0),
    _mala_init,
    prior_fn(),
    lp,
    n_chains=8,
    n_samples=200,
    n_warmup=100,
  )
  chex.assert_shape(samples["mean"], (8, 100, 2))
  chex.assert_shape(samples["std"], (8, 100, 1))
  assert isinstance(info, MCMCSampleInfo)


def test_sample_with_nuts_returns_samples_and_mcmc_info():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def lp(theta):
    return jnp.sum(prior.log_prob(theta))

  samples, info = sample_with_nuts(
    jr.PRNGKey(0), lp, prior, n_chains=2, n_samples=40, n_warmup=20
  )
  assert samples["theta"].shape == (2, 20, 2)
  assert isinstance(info, MCMCSampleInfo)
  assert jnp.isfinite(info.acceptance_rate)
