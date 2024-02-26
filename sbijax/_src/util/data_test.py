# pylint: skip-file

import chex
import distrax
import haiku as hk
from jax import numpy as jnp
from jax import random as jr
from surjectors import Chain, MaskedCoupling, TransformedDistribution
from surjectors.nn import make_mlp
from surjectors.util import make_alternating_binary_mask

from sbijax._src.snl import SNL
from sbijax._src.util.data import stack_data


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


def make_model(dim):
    def _bijector_fn(params):
        means, log_scales = jnp.split(params, 2, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(method, **kwargs):
        layers = []
        for i in range(2):
            mask = make_alternating_binary_mask(dim, i % 2 == 0)
            layer = MaskedCoupling(
                mask=mask,
                bijector_fn=_bijector_fn,
                conditioner=make_mlp([8, 8, dim * 2]),
            )
            layers.append(layer)
        chain = Chain(layers)
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(dim), jnp.ones(dim)),
            1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


def test_stack_data():
    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    snl = SNL(fns, make_model(2))
    n = 100
    data, _ = snl.simulate_data(jr.PRNGKey(1), n_simulations=n)
    also_data, _ = snl.simulate_data(jr.PRNGKey(2), n_simulations=n)
    stacked_data = stack_data(data, also_data)

    chex.assert_trees_all_equal(data[0], stacked_data[0][:n])
    chex.assert_trees_all_equal(data[1], stacked_data[1][:n])
    chex.assert_trees_all_equal(also_data[0], stacked_data[0][n:])
    chex.assert_trees_all_equal(also_data[1], stacked_data[1][n:])


def test_stack_data_with_none():
    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    snl = SNL(fns, make_model(2))
    n = 100
    data, _ = snl.simulate_data(jr.PRNGKey(1), n_simulations=n)
    stacked_data = stack_data(None, data)

    chex.assert_trees_all_equal(data[0], stacked_data[0])
    chex.assert_trees_all_equal(data[1], stacked_data[1])
