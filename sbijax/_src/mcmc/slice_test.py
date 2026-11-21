import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference._sample_info import MCMCSampleInfo
from sbijax._src.mcmc.slice import sample_with_slice
from sbijax.mcmc import make_sampler, slice_sampling

prior = tfd.JointDistributionNamed(
  {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
)


def lp(theta):
  return jnp.sum(prior.log_prob(theta))


@pytest.mark.parametrize("make_key", [jr.PRNGKey, jr.key])
def test_sample_with_slice_returns_samples_and_info(make_key):
  samples, info = sample_with_slice(
    make_key(0), lp, prior, n_chains=2, n_samples=20, n_warmup=20
  )
  assert samples["theta"].shape == (2, 20, 2)
  assert isinstance(info, MCMCSampleInfo)
  assert 0.0 <= float(info.acceptance_rate) <= 1.0  # nan under TFP


def test_slice_recovers_conjugate_posterior(conjugate_model):
  prior_fn, log_prob, posterior_mean = conjugate_model
  samples, info = sample_with_slice(
    jr.PRNGKey(1),
    log_prob,
    prior_fn(),
    n_chains=4,
    n_samples=1_000,
    n_warmup=500,
  )
  for k, v in posterior_mean.items():
    assert abs(float(samples[k].mean()) - v) < 0.1
  assert max(float(r.max()) for r in jax.tree.leaves(info.rhat)) < 1.05


def test_slice_stays_in_constrained_support(prior_log_prob_tuple):
  prior_fn, log_prob = prior_log_prob_tuple
  samples, _ = sample_with_slice(
    jr.PRNGKey(1),
    log_prob,
    prior_fn(),
    n_chains=10,
    n_samples=100,
    n_warmup=100,
  )
  chex.assert_shape(samples["mean"], (10, 100, 2))
  chex.assert_shape(samples["std"], (10, 100, 1))
  assert bool(jnp.all(samples["std"] > 0))
  assert bool(jnp.all(jnp.isfinite(samples["mean"])))


def test_single_chain_without_thinning():
  samples, info = sample_with_slice(
    jr.PRNGKey(0), lp, prior, n_chains=1, n_samples=30, n_warmup=10, n_thin=0
  )
  assert samples["theta"].shape == (1, 30, 2)
  assert info.rhat is None and info.ess is None


def test_unknown_kernel_option_raises():
  with pytest.raises(TypeError):
    sample_with_slice(jr.PRNGKey(0), lp, prior, n_doublings=3)


@pytest.mark.parametrize("bad", [{"n_thin": -1}, {"step_size": 0.0}])
def test_invalid_kernel_options_raise(bad):
  with pytest.raises(ValueError):
    sample_with_slice(jr.PRNGKey(0), lp, prior, n_samples=5, n_warmup=5, **bad)


def test_make_sampler_with_slice_sampling_recovers_posterior(conjugate_model):
  prior_fn, _, posterior_mean = conjugate_model

  def loglik(theta):
    return tfd.Normal(theta["a"], 1.0).log_prob(1.5) + tfd.Normal(
      theta["b"], 1.0
    ).log_prob(-0.5)

  sampler = make_sampler(slice_sampling, prior=prior_fn(), n_thin=1)
  samples, info = sampler(
    jr.PRNGKey(2), loglik, n_chains=4, n_samples=1_000, n_warmup=500
  )
  for k, v in posterior_mean.items():
    assert abs(float(samples[k].mean()) - v) < 0.1
  assert 0.0 <= float(info.acceptance_rate) <= 1.0
