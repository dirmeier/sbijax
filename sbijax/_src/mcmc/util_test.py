# pylint: skip-file

import blackjax as bj
import chex
import jax
from jax import random as jr

from sbijax._src.mcmc.util import run_blackjax


def _mala_init(rng_key, n_chains, prior, lp):
  initial_positions = prior.sample(seed=rng_key, sample_shape=(n_chains,))
  kernel = bj.mala(lp, 0.1)
  initial_state = jax.vmap(kernel.init)(initial_positions)
  return initial_state, kernel.step


def test_run_blackjax_returns_chain_shaped_samples(prior_log_prob_tuple):
  prior_fn, lp = prior_log_prob_tuple
  samples = run_blackjax(
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
