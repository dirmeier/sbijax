"""
Example using ABC
"""

import distrax
import jax
import matplotlib.pyplot as plt
import seaborn as sns
from jax import numpy as jnp

from sbijax import SMCABC


def prior_model_fns(leng):
    p = distrax.Independent(
        distrax.Uniform(jnp.full(leng, -2.0), jnp.full(leng, 2.0)), 1
    )
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.MultivariateNormalDiag(theta, 0.1 * jnp.ones_like(theta))
    y = p.sample(seed=seed)
    return y


def summary_fn(y):
    if y.ndim == 2:
        y = y[None, ...]
    sumr = jnp.mean(y, axis=1, keepdims=True)
    return sumr


def distance_fn(y_simulated, y_observed):
    diff = y_simulated - y_observed
    dist = jax.vmap(lambda el: jnp.linalg.norm(el))(diff)
    return dist


def run():
    len_thetas = 2
    y_observed = jnp.ones(len_thetas)

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns(len_thetas)
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    smc = SMCABC(fns, summary_fn, distance_fn)
    smc.fit(23, y_observed)
    smc_samples, _ = smc.sample_posterior(10, 1000, 1000, 0.75, 500)

    fig, axes = plt.subplots(len_thetas)
    for i in range(len_thetas):
        sns.histplot(smc_samples[:, i], color="darkblue", ax=axes[i])
        axes[i].set_title(rf"Approximated posterior $\theta_{i}$")
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
