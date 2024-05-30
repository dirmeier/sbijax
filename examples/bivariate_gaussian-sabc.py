"""
Example using sequential neural likelihood estimation on a bivariate Gaussian
"""

import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import NLE, plot_posterior
from sbijax.nn import make_mdn


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        mean=tfd.Normal(jnp.zeros(2), 1.0),
        scale=tfd.HalfNormal(jnp.ones(1)),
    ), batch_ndims=0)
    return prior


def simulator_fn(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["mean"]), 1.0)
    y = theta["mean"] + theta["scale"] * p.sample(seed=seed)
    return y


def run():
    y_observed = jnp.array([-2.0, 1.0])
    fns = prior_fn, simulator_fn
    model = NLE(fns, make_mdn(2, 10))

    data, _ = model.simulate_data(jr.PRNGKey(11))
    params, info = model.fit(jr.PRNGKey(2), data=data, n_early_stopping_patience=25)
    inference_result, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)

    plot_posterior(inference_result)
    plt.show()


if __name__ == "__main__":
    run()
