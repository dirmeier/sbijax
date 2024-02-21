"""
Example using sequential posterior estimation  on a bivariate Gaussian
"""

import distrax
import haiku as hk
import jax.nn
import matplotlib.pyplot as plt
import optax
import seaborn as sns
from jax import numpy as jnp
from jax import random as jr
from surjectors import (
    Chain,
    MaskedAutoregressive,
    Permutation,
    TransformedDistribution,
)
from surjectors.nn import MADE
from surjectors.util import unstack

from sbijax import SNP


def prior_model_fns():
    p = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.Normal(jnp.zeros_like(theta), 1.0)
    y = theta + p.sample(seed=seed)
    return y


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
    return td


def run():
    y_observed = jnp.array([-2.0, 1.0])

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    snp = SNP(fns, make_model(2))
    optimizer = optax.adam(1e-3)

    data, params = None, {}
    for i in range(2):
        data, _ = snp.simulate_data_and_possibly_append(
            jr.fold_in(jr.PRNGKey(1), i),
            params=params,
            observable=y_observed,
            data=data,
        )
        params, info = snp.fit(
            jr.fold_in(jr.PRNGKey(2), i),
            data=data,
            optimizer=optimizer,
        )

    rng_key = jr.PRNGKey(23)
    snp_samples, _ = snp.sample_posterior(rng_key, params, y_observed)
    fig, axes = plt.subplots(2)
    for i, ax in enumerate(axes):
        sns.histplot(snp_samples[:, i], color="darkblue", ax=ax)
        ax.set_xlim([-3.0, 3.0])
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
