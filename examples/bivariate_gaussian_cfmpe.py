"""
Example using consistency model posterior estimation on a bivariate Gaussian
"""

import distrax
import haiku as hk
import matplotlib.pyplot as plt
import optax
import seaborn as sns
from jax import numpy as jnp
from jax import random as jr

from sbijax import SCMPE
from sbijax.nn import ConsistencyModel


def prior_model_fns():
    p = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.Normal(jnp.zeros_like(theta), 1.0)
    y = theta + p.sample(seed=seed)
    return y


def make_model(dim):
    @hk.transform
    def _mlp(method, **kwargs):
        def _c_skip(time):
            return 1 / ((time - 0.001) ** 2 + 1)

        def _c_out(time):
            return 1.0 * (time - 0.001) / jnp.sqrt(1 + time ** 2)
        def _nn(theta, time, context, **kwargs):
            ins = jnp.concatenate([theta, time, context], axis=-1)
            outs = hk.nets.MLP([64, 64, dim])(ins)
            out_skip = _c_skip(time) * theta + _c_out(time) * outs
            return out_skip

        cm = ConsistencyModel(dim, _nn)
        return cm(method, **kwargs)

    return _mlp


def run():
    y_observed = jnp.array([2.0, -2.0])

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    estim = SCMPE(fns, make_model(2))
    optimizer = optax.adam(1e-3)

    data, params = None, {}
    for i in range(2):
        data, _ = estim.simulate_data_and_possibly_append(
            jr.fold_in(jr.PRNGKey(1), i),
            params=params,
            observable=y_observed,
            data=data,
        )
        params, info = estim.fit(
            jr.fold_in(jr.PRNGKey(2), i),
            data=data,
            optimizer=optimizer,
        )


    rng_key = jr.PRNGKey(23)
    post_samples, _ = estim.sample_posterior(rng_key, params, y_observed)
    print(post_samples)
    fig, axes = plt.subplots(2)
    for i, ax in enumerate(axes):
        sns.histplot(post_samples[:, i], color="darkblue", ax=ax)
        ax.set_xlim([-3.0, 3.0])
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
