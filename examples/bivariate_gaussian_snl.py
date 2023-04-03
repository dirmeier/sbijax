"""
Example using SNL and masked coupling flows
"""

from functools import partial

import distrax
import haiku as hk
import matplotlib.pyplot as plt
import optax
import seaborn as sns
from jax import numpy as jnp
from jax import random
from surjectors import Chain, MaskedCoupling, TransformedDistribution
from surjectors.conditioners import mlp_conditioner
from surjectors.util import make_alternating_binary_mask

from sbijax import SNL
from sbijax.mcmc import sample_with_nuts


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


def run():
    rng_seq = hk.PRNGSequence(0)
    y_observed = jnp.array([-1.0, 1.0])

    log_density_partial = partial(log_density_fn, y=y_observed)
    log_density = lambda x: log_density_partial(**x)

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    snl = SNL(fns, make_model(2))
    optimizer = optax.adam(1e-3)
    params, info = snl.fit(
        random.PRNGKey(23),
        y_observed,
        n_rounds=1,
        optimizer=optimizer,
        sampler="slice",
    )

    nuts_samples = sample_with_nuts(rng_seq, log_density, 2, 4, 2000, 1000)
    snl_samples, _ = snl.sample_posterior(
        params, 4, 10000, 7500, sampler="slice"
    )

    snl_samples = snl_samples.reshape(-1, 2)
    nuts_samples = nuts_samples.reshape(-1, 2)

    fig, axes = plt.subplots(2, 2)
    for i in range(2):
        sns.histplot(nuts_samples[:, i], color="darkgrey", ax=axes.flatten()[i])
        sns.histplot(
            snl_samples[:, i], color="darkblue", ax=axes.flatten()[i + 2]
        )
        axes.flatten()[i].set_title(rf"Sampled posterior $\theta_{i}$")
        axes.flatten()[i + 2].set_title(
            rf"Approximated posterior $\theta_{i + 2}$"
        )
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
