# pylint: skip-file

from jax import numpy as jnp
from jax import random as jr

from sbijax.experimental import AiO
from sbijax.experimental.nn import make_simformer_based_score_model


def test_aio(prior_simulator_tuple):
    y_observed = jnp.array([-1.0, 1.0])
    estim = AiO(prior_simulator_tuple, make_simformer_based_score_model(2, jnp.eye(4)))
    data, params = None, {}
    for i in range(2):
        data, _ = estim.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
            data=data,
            n_simulations=100,
        )
        params, info = estim.fit(jr.PRNGKey(2), data=data, n_iter=2)
    _ = estim.sample_posterior(
        jr.PRNGKey(3),
        params,
        y_observed,
    )
