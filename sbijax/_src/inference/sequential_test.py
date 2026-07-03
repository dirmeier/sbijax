# pylint: skip-file

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.posterior.npe import NPEInfo, npe
from sbijax._src.inference.sequential import run_sequential
from sbijax._src.nn.make_flow import make_maf


def _gaussian_problem():
  # theta ~ N(0, 1); y = theta + N(0, 1). For a single observation the
  # posterior is N(y_obs / 2, 1 / 2) in each dimension.
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


def test_run_sequential_advances_rounds():
  prior, simulator = _gaussian_problem()
  est = npe(prior, make_maf(2))
  _, info = run_sequential(
    jr.PRNGKey(0),
    est,
    prior,
    simulator,
    jnp.zeros(2),
    n_rounds=2,
    n_simulations_per_round=200,
    n_iter=2,
    batch_size=100,
  )
  assert isinstance(info, NPEInfo)
  assert info.round == 1
  assert info.num_atoms == 10
  assert info.losses.ndim == 2 and info.losses.shape[1] == 2


def test_run_sequential_recovers_gaussian_posterior_mean():
  prior, simulator = _gaussian_problem()
  y_obs = jnp.array([1.5, -1.5])
  analytic_mean = y_obs / 2.0
  est = npe(prior, make_maf(2))
  params, _ = run_sequential(
    jr.PRNGKey(0),
    est,
    prior,
    simulator,
    y_obs,
    n_rounds=2,
    n_simulations_per_round=2000,
    n_iter=1000,
    batch_size=100,
  )
  samples, _ = est.sample(jr.PRNGKey(1), params, y_obs, n_samples=2000)
  post_mean = jnp.mean(samples["theta"].reshape(-1, 2), axis=0)
  # a correct atomic round recovers a posterior near the analytic mean, and
  # much closer than the prior mean (0) is -- a gross event-space or sign bug
  # in the atomic loss would fail this.
  assert jnp.linalg.norm(post_mean - analytic_mean) < 0.4
  assert jnp.linalg.norm(post_mean - analytic_mean) < jnp.linalg.norm(
    analytic_mean
  )
