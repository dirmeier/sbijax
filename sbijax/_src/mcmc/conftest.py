# pylint: skip-file

import pytest
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


def prior_fn():
  prior = tfd.JointDistributionNamed(
    {
      "mean": tfd.Normal(jnp.zeros(2), jnp.array(1.0)),
      "std": tfd.HalfNormal(jnp.array(1.0)),
    },
    batch_ndims=0,
  )
  return prior


def log_prob(theta):
  y = jnp.array([-2.0, 2.0])
  lp_prior = prior_fn().log_prob(theta)
  lp_data = tfd.Normal(theta["mean"], theta["std"]).log_prob(y)
  return jnp.sum(lp_data) + jnp.sum(lp_prior)


@pytest.fixture()
def prior_log_prob_tuple(request):
  yield prior_fn, log_prob


def conjugate_prior_fn():
  prior = tfd.JointDistributionNamed(
    {"a": tfd.Normal(0.0, 1.0), "b": tfd.Normal(0.0, 1.0)},
    batch_ndims=0,
  )
  return prior


def conjugate_log_prob(theta):
  lp_data = tfd.Normal(theta["a"], 1.0).log_prob(1.5)
  lp_data += tfd.Normal(theta["b"], 1.0).log_prob(-0.5)
  return lp_data + conjugate_prior_fn().log_prob(theta)


@pytest.fixture()
def conjugate_model(request):
  # two scalar leaves with a normal likelihood: the posterior means are y / 2
  yield conjugate_prior_fn, conjugate_log_prob, {"a": 0.75, "b": -0.25}
