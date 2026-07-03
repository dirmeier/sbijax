"""Contract test for the cmpe ObjectiveFns factory."""

import jax.numpy as jnp
import optax
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.experimental.cmpe import cmpe
from sbijax._src.nn.make_consistency_model import make_cm
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train._types import ObjectiveFns
from sbijax._src.train.fit import fit
from sbijax._src.train.sample import sample


def test_cmpe_objective_trains_and_samples():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def sim(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  obj = cmpe(make_cm(2))
  assert isinstance(obj, ObjectiveFns)
  data = simulate(jr.key(0), prior, sim, n=200)
  params, _ = fit(
    jr.key(1), obj, data, optimizer=optax.adam(3e-4), n_iter=2, batch_size=100
  )
  samples, _ = sample(jr.key(2), obj, params, jnp.zeros(2), n_samples=64)
  assert samples["theta"].shape == (1, 64, 2)
