"""
Example using SNL and masked autoregressive flows flows
"""

from functools import partial

import distrax
import haiku as hk
import jax
import matplotlib.pyplot as plt
import optax
import seaborn as sns
from jax import numpy as jnp
from jax import random
from surjectors import (
    Chain,
    MaskedAutoregressive,
    Permutation,
    TransformedDistribution,
)
from surjectors.conditioners import MADE
from surjectors.util import unstack

from sbijax import SNL
from sbijax.mcmc import sample_with_slice


def prior_model_fns():
    p = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.Normal(jnp.zeros_like(theta), 0.1)
    y = theta + p.sample(seed=seed)
    return y


def log_density_fn(theta, y):
    prior = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    likelihood = distrax.MultivariateNormalDiag(
        theta, 0.1 * jnp.ones_like(theta)
    )

    lp = jnp.sum(prior.log_prob(theta)) + jnp.sum(likelihood.log_prob(y))
    return lp


def make_model(dim):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(method, **kwargs):
        layers = []
        order = jnp.arange(dim)
        for i in range(5):
            layer = MaskedAutoregressive(
                bijector_fn=_bijector_fn,
                conditioner=MADE(
                    dim,
                    [50, dim * 2],
                    2,
                    w_init=hk.initializers.TruncatedNormal(0.001),
                    b_init=jnp.zeros,
                    activation=jax.nn.tanh,
                ),
            )
            order = order[::-1]
            layers.append(layer)
            layers.append(Permutation(order, 1))
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
    y_observed = jnp.array([-2.0, 1.0])

    log_density_partial = partial(log_density_fn, y=y_observed)
    log_density = lambda x: jax.vmap(log_density_partial)(x)

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    snl = SNL(fns, make_model(2))
    optimizer = optax.adam(1e-3)
    params, info = snl.fit(
        random.PRNGKey(23),
        y_observed,
        optimizer=optimizer,
        n_rounds=3,
        max_n_iter=100,
        batch_size=64,
        n_early_stopping_patience=5,
        sampler="slice",
    )

    slice_samples = sample_with_slice(
        hk.PRNGSequence(0), log_density, 4, 2000, 1000, prior_simulator_fn
    )
    slice_samples = slice_samples.reshape(-1, 2)
    snl_samples, _ = snl.sample_posterior(
        params, 4, 2000, 1000, sampler="slice"
    )

    print(f"Took n={snl.n_total_simulations} simulations in total")
    fig, axes = plt.subplots(2, 2)
    for i in range(2):
        sns.histplot(
            slice_samples[:, i], color="darkgrey", ax=axes.flatten()[i]
        )
        sns.histplot(
            snl_samples[:, i], color="darkblue", ax=axes.flatten()[i + 2]
        )
        axes.flatten()[i].set_title(rf"Sampled posterior $\theta_{i}$")
        axes.flatten()[i + 2].set_title(rf"Approximated posterior $\theta_{i}$")
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
