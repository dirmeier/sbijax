# pylint: skip-file

# A small sbibm-style benchmark (DR-009): on a reference task with a known
# posterior, check that a couple of method families recover its mean and
# covariance. The task is conjugate Gaussian -- theta ~ N(0, I), y | theta ~
# N(theta, I) -- so the posterior for a single observation is
# N(y / 2, (1 / 2) I).

import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.inference.posterior.npe import npe
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.simulate import simulate


def _gaussian_problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


BENCHMARK = {
  "npe": {
    "build": lambda prior: npe(prior, make_maf(2)),
    "sample_kwargs": {"n_samples": 2_000},
  },
  "nle": {
    "build": lambda prior: nle(prior, make_maf(2)),
    # sbijax MCMC treats n_samples as total (warmup + kept), so keep 1000/chain
    "sample_kwargs": {"n_chains": 4, "n_samples": 2_000, "n_warmup": 1_000},
  },
}


@pytest.mark.parametrize("name", list(BENCHMARK))
def test_recovers_gaussian_posterior(name):
  prior, simulator = _gaussian_problem()
  y_obs = jnp.array([1.0, -2.0])
  post_mean = y_obs / 2.0
  post_std = jnp.sqrt(0.5)

  data = simulate(jr.PRNGKey(0), prior, simulator, n=5_000)
  est = BENCHMARK[name]["build"](prior)
  params, _ = est.fit(jr.PRNGKey(1), data, n_iter=1_000, batch_size=100)
  idata = est.sample(
    jr.PRNGKey(2), params, y_obs, **BENCHMARK[name]["sample_kwargs"]
  )
  theta = idata["/posterior"]["theta"].data.reshape(-1, 2)

  assert jnp.linalg.norm(jnp.mean(theta, axis=0) - post_mean) < 0.3
  assert jnp.all(jnp.abs(jnp.std(theta, axis=0) - post_std) < 0.2)
