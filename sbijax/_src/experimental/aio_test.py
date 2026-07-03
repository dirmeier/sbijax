# pylint: skip-file

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.experimental.aio import aio
from sbijax._src.experimental.nn.make_simformer import (
  make_simformer_based_score_model,
)
from sbijax._src.simulate.simulate import simulate


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


def test_aio_fit_then_sample():
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=64)
  est = aio(prior, make_simformer_based_score_model(2, jnp.eye(4), 1, 1))
  params, _ = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=32)
  samples, _ = est.sample(jr.PRNGKey(2), params, jnp.zeros(2), n_samples=16)
  theta = samples["theta"]
  assert theta.ndim == 3 and theta.shape[-1] == 2
