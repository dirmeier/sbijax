# pylint: skip-file
import chex
from jax import random as jr

from sbijax._src.mcmc.rmh import sample_with_rmh


def test_rmh_sampler(prior_log_prob_tuple):
  samples, _ = sample_with_rmh(
    jr.PRNGKey(1),
    prior_log_prob_tuple[1],
    prior_log_prob_tuple[0](),
    n_chains=10,
    n_samples=100,
    n_warmup=100,
  )
  chex.assert_shape(samples["mean"], (10, 100, 2))
  chex.assert_shape(samples["std"], (10, 100, 1))


def test_rmh_recovers_conjugate_posterior(conjugate_model):
  prior_fn, log_prob, posterior_mean = conjugate_model
  samples, _ = sample_with_rmh(
    jr.PRNGKey(1),
    log_prob,
    prior_fn(),
    n_chains=4,
    n_samples=4_000,
    n_warmup=1_000,
  )
  for k, v in posterior_mean.items():
    assert abs(float(samples[k].mean()) - v) < 0.1
