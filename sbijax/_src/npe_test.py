# pylint: skip-file

from jax import numpy as jnp, random as jr

from sbijax import NPE
from sbijax.nn import make_maf


def test_npe(prior_simulator_tuple):
    y_observed = jnp.array([-1.0, 1.0])
    snp = NPE(prior_simulator_tuple, make_maf(2))
    data, params = None, {}
    for i in range(2):
        data, _ = snp.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params,
            observable=y_observed,
            data=data,
            n_simulations=100,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        params, info = snp.fit(jr.PRNGKey(3), data=data, n_iter=2)
    _ = snp.sample_posterior(
        jr.PRNGKey(3),
        params,
        y_observed,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
