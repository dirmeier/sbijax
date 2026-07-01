# pylint: skip-file

import chex
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.facade.nle import NLE
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.simulate import simulate


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


def test_facade_fit_then_sample_without_passing_params():
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=200)
  est = NLE(prior, make_maf(2))
  params, info = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=100)
  assert params is not None
  # facade holds params: sample() needs only an observation
  idata = est.sample(
    jr.PRNGKey(2), jnp.zeros(2), n_chains=2, n_samples=30, n_warmup=10
  )
  chex.assert_shape(idata["/posterior"]["theta"].data, (2, 20, 2))
