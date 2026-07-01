# pylint: skip-file

import chex
import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.simulate import simulate
from sbijax.nn import make_maf


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


# registry of trainable estimators to check against the Estimator contract
ESTIMATORS = {
  "nle": lambda prior: nle(prior, make_maf(2)),
}


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_fit_returns_params_and_loss_history(name):
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=200)
  est = ESTIMATORS[name](prior)
  params, info = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=100)
  assert params is not None
  chex.assert_shape(info, (2, 2))


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_sample_returns_chain_shaped_inference_data(name):
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=200)
  est = ESTIMATORS[name](prior)
  params, _ = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=100)
  idata = est.sample(
    jr.PRNGKey(2), params, jnp.zeros(2), n_chains=2, n_samples=30, n_warmup=10
  )
  theta = idata["/posterior"]["theta"].data
  chex.assert_shape(theta, (2, 20, 2))
