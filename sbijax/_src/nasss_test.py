# pylint: skip-file

from jax import numpy as jnp, random as jr

from sbijax import NASSS, NLE
from sbijax.nn import make_nasss_net, make_maf
from tensorflow_probability.substrates.jax import distributions as tfd


def simulator_fn(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["theta"]), 0.1)
    y = theta["theta"] + p.sample(seed=seed)
    y = jnp.tile(y, (1, 5))
    return y


def test_nasss(prior_simulator_tuple):
    y_observed = jr.normal(jr.PRNGKey(0), (10,))
    fns = prior_simulator_tuple[0], simulator_fn

    model_nass = NASSS(
        fns,
        make_nasss_net((32, 5), (32, 1), (32, 1)),
    )
    model_nle = NLE(fns, make_maf(5))

    data, params_nle, params_nass = None, {}, {}
    for i in range(2):
        s_observed = model_nass.summarize(params_nass, y_observed)
        data, _ = model_nle.simulate_data_and_possibly_append(
            jr.PRNGKey(1),
            params=params_nle,
            observable=s_observed,
            data=data,
            n_simulations=100,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        params_nass, _ = model_nass.fit(jr.PRNGKey(2), data=data, n_iter=2)
        summaries = model_nass.summarize(params_nass, data)
        params_nle, _ = model_nle.fit(jr.PRNGKey(3), data=summaries, n_iter=2)
    s_observed = model_nass.summarize(params_nass, y_observed)
    _ = model_nle.sample_posterior(
        jr.PRNGKey(3),
        params_nle,
        s_observed,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
