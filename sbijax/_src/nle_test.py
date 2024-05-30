# pylint: skip-file

import pytest
from jax import numpy as jnp
from jax import random as jr

from sbijax import NLE
from sbijax.nn import make_maf


def test_snl(prior_simulator_tuple):
    y_observed = jnp.array([-1.0, 1.0])
    snl = NLE(prior_simulator_tuple, make_maf(2))
    data, params = None, {}
    for i in range(2):
        data, _ = snl.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
            data=data,
            n_simulations=100,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        params, info = snl.fit(jr.PRNGKey(2), data=data, n_iter=2)
    _ = snl.sample_posterior(
        jr.PRNGKey(3),
        params,
        y_observed,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )


def test_simulate_data_from_posterior_fail(prior_simulator_tuple):
    snl = NLE(prior_simulator_tuple, make_maf(2))
    n = 100

    data, _ = snl.simulate_data(jr.PRNGKey(1), n_simulations=n)
    params, _ = snl.fit(jr.PRNGKey(2), data=data, n_iter=10)
    with pytest.raises(ValueError):
        snl.simulate_data(jr.PRNGKey(3), n_simulations=n, params=params)
