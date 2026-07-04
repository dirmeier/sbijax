"""Tests for the npe ObjectiveFns API."""

import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.posterior.npe import npe
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train._types import ObjectiveFns
from sbijax._src.train.sample import sample
from sbijax._src.train.train import train


def test_npe_objective_amortized_and_atomic():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def sim(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  obj = npe(make_maf(2))
  assert isinstance(obj, ObjectiveFns)
  assert isinstance(obj.extra(prior), ObjectiveFns)
  data = simulate(jr.key(0), prior, sim, n=200)
  params, _ = train(jr.key(1), obj, data, n_iter=2, batch_size=100)
  params, _ = train(jr.key(2), obj.extra(prior), data, n_iter=2, batch_size=100)
  samples, _ = sample(jr.key(3), obj, params, jnp.zeros(2), n_samples=64)
  assert samples["theta"].shape == (1, 64, 2)
