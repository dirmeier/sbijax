# pylint: skip-file

import distrax
import haiku as hk
from jax import numpy as jnp
from jax import random as jr

from sbijax import CMPE
from sbijax.nn import make_cm


def test_cmpe(prior_simulator_tuple):
    y_observed = jnp.array([-1.0, 1.0])
    estim = CMPE(prior_simulator_tuple, make_cm(2))
    data, params = None, {}
    for i in range(2):
        data, _ = estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
            data=data,
            n_simulations=100,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        params, info = estim.fit(jr.PRNGKey(2), data=data, n_iter=2)
    _ = estim.sample_posterior(
        jr.PRNGKey(3),
        params,
        y_observed,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
