"""
Example using sequential Monte Carlo ABC on a bivariate Gaussian
"""

import distrax
import jax
import matplotlib.pyplot as plt
import seaborn as sns
from jax import numpy as jnp
from jax import random as jr

from sbijax import SMCABC


def prior_model_fns():
    p = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.Normal(jnp.zeros_like(theta), 0.1)
    y = theta + p.sample(seed=seed)
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
    y_observed = jnp.array([-2.0, 1.0])

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    smc = SMCABC(fns, summary_fn, distance_fn)
    smc_samples, _ = smc.sample_posterior(
        jr.PRNGKey(22), y_observed, 10, 1000, 1000, 0.6, 500
    )

    fig, axes = plt.subplots(2)
    for i in range(2):
        sns.histplot(smc_samples[:, i], color="darkblue", ax=axes[i])
        axes[i].set_title(rf"Approximated posterior $\theta_{i}$")
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
