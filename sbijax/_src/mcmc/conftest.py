# pylint: skip-file

import pytest
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


def prior_fn():
    prior = tfd.JointDistributionNamed(
        dict(mean=tfd.Normal(jnp.zeros(2), 1.0), std=tfd.HalfNormal(1.0)),
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
