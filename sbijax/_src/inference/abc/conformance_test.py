# pylint: skip-file

import jax
import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.abc.sabc import sabc
from sbijax._src.inference.abc.smcabc import smcabc


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 3.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 0.1).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


def _l2(x, y):
  return jax.vmap(jnp.linalg.norm)(x - y)


# each entry builds an ABCSampler and provides small sampling budgets
ABC_SAMPLERS = {
  "sabc": {
    "build": sabc,
    "sample_kwargs": {"n_particles": 64, "n_simulation": 2_000},
  },
  "smcabc": {
    "build": lambda prior, sim: smcabc(prior, sim, lambda x: x, _l2),
    "sample_kwargs": {"n_rounds": 2, "n_particles": 200},
  },
}


@pytest.mark.parametrize("name", list(ABC_SAMPLERS))
def test_abc_sample_returns_posterior_inference_data(name):
  prior, simulator = _problem()
  sampler = ABC_SAMPLERS[name]["build"](prior, simulator)
  idata = sampler.sample(
    jr.PRNGKey(0), jnp.zeros(2), **ABC_SAMPLERS[name]["sample_kwargs"]
  )
  theta = idata["/posterior"]["theta"].data
  assert theta.ndim == 3 and theta.shape[-1] == 2
