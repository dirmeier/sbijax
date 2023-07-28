# pylint: skip-file

import distrax
import haiku as hk
import optax
from jax import numpy as jnp
from surjectors import Chain, MaskedCoupling, TransformedDistribution
from surjectors.conditioners import mlp_conditioner
from surjectors.util import make_alternating_binary_mask

from sbijax import SNL


def prior_model_fns():
    p = distrax.Independent(
        distrax.Uniform(jnp.full(2, -3.0), jnp.full(2, 3.0)), 1
    )
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
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([8, 8, dim * 2]),
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


def test_snl():
    rng_seq = hk.PRNGSequence(0)
    y_observed = jnp.array([-1.0, 1.0])

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    snl = SNL(fns, make_model(2))
    params, info = snl.fit(
        next(rng_seq),
        y_observed,
        n_rounds=1,
        optimizer=optax.adam(1e-4),
        sampler="slice",
    )
    _ = snl.sample_posterior(params, 2, 100, 50, sampler="slice")
