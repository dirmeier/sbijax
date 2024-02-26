"""
Example using sequential neural ratio estimation on a bivariate Gaussian
"""

import distrax
import haiku as hk
import matplotlib.pyplot as plt
import optax
import seaborn as sns
from jax import numpy as jnp
from jax import random as jr

from sbijax import SNR


def prior_model_fns():
    p = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.Normal(jnp.zeros_like(theta), 1.0)
    y = theta + p.sample(seed=seed)
    return y


def make_model():
    @hk.without_apply_rng
    @hk.transform
    def _mlp(inputs, **kwargs):
        return hk.nets.MLP([64, 64, 1])(inputs)

    return _mlp


def run():
    y_observed = jnp.array([2.0, -2.0])

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    snr = SNR(fns, make_model())
    optimizer = optax.adam(1e-3)

    data, params = None, {}
    for i in range(2):
        data, _ = snr.simulate_data_and_possibly_append(
            jr.fold_in(jr.PRNGKey(1), i),
            params=params,
            observable=y_observed,
            data=data,
        )
        params, info = snr.fit(
            jr.fold_in(jr.PRNGKey(2), i),
            data=data,
            optimizer=optimizer,
        )

    rng_key = jr.PRNGKey(23)
    snr_samples, _ = snr.sample_posterior(rng_key, params, y_observed)
    fig, axes = plt.subplots(2)
    for i, ax in enumerate(axes):
        sns.histplot(snr_samples[:, i], color="darkblue", ax=ax)
        ax.set_xlim([-3.0, 3.0])
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
