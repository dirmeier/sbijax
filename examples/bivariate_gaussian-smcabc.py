"""Sequential Monte Carlo ABC example.

Demonstrates sequential Monte Carlo ABC on a simple bivariate Gaussian example.
"""

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import SMCABC, plot_posterior


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Normal(jnp.zeros(2), jnp.ones(2))
    ), batch_ndims=0)
    return prior


def simulator_fn(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["theta"]), 0.1)
    y = theta["theta"] + p.sample(seed=seed)
    return y


def summary_fn(y):
    return y


def distance_fn(y_simulated, y_observed):
    diff = y_simulated - y_observed
    dist = jax.vmap(lambda el: jnp.linalg.norm(el))(diff)
    return dist


def run():
    y_observed = jnp.array([-2.0, 1.0])

    fns = prior_fn, simulator_fn

    smc = SMCABC(fns, summary_fn, distance_fn)
    smc_samples, _ = smc.sample_posterior(
        jr.PRNGKey(1), y_observed, 10, 1000, 0.85, 500
    )
    plot_posterior(smc_samples)
    plt.show()


if __name__ == "__main__":
    run()
