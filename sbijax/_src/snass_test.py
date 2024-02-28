# pylint: skip-file

import distrax
import haiku as hk
from jax import numpy as jnp

from sbijax import SNASS
from sbijax.nn import make_affine_maf, make_snass_net


def prior_model_fns():
    p = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.MultivariateNormalDiag(theta, 0.1 * jnp.ones_like(theta))
    y = p.sample(seed=seed)
    return y


def log_density_fn(theta, y):
    prior = distrax.Uniform(jnp.full(2, -3.0), jnp.full(2, 3.0))
    likelihood = distrax.MultivariateNormalDiag(
        theta, 0.1 * jnp.ones_like(theta)
    )

    lp = jnp.sum(prior.log_prob(theta)) + jnp.sum(likelihood.log_prob(y))
    return lp


def test_snass():
    rng_seq = hk.PRNGSequence(0)
    y_observed = jnp.array([-1.0, 1.0])

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    estim = SNASS(
        fns, make_affine_maf(1, 2, (32, 32)), make_snass_net((32, 1), (32, 1))
    )
    data, params = None, {}
    for i in range(2):
        data, _ = estim.simulate_data_and_possibly_append(
            next(rng_seq),
            params=params,
            observable=y_observed,
            data=data,
            n_simulations=100,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
        )
        params, info = estim.fit(next(rng_seq), data=data, n_iter=2)
    _ = estim.sample_posterior(
        next(rng_seq),
        params,
        y_observed,
        n_chains=2,
        n_samples=200,
        n_warmup=100,
    )
