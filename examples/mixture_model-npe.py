"""Neural posterior estimation example.

Demonstrates NPE on a simple mixture model.
"""
import argparse

import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import plot_posterior, NPE
from sbijax.nn import make_maf


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Normal(jnp.zeros(2), 1)
    ), batch_ndims=0)
    return prior


def simulator_fn(seed, theta):
    mean = theta["theta"].reshape(-1, 2)
    n = mean.shape[0]
    data_key, cat_key = jr.split(seed)
    categories = tfd.Categorical(logits=jnp.zeros(2)).sample(seed=cat_key, sample_shape=(n,))
    scales = jnp.array([1.0, 0.1])[categories].reshape(-1, 1)
    y = tfd.Normal(mean, scales).sample(seed=data_key)
    return y


def run(n_iter):
    y_observed = jnp.array([-2.0, 1.0])
    fns = prior_fn, simulator_fn
    neural_network = make_maf(2)
    model = NPE(fns, neural_network, use_event_space_bijections=False)

    data, _ = model.simulate_data(jr.PRNGKey(1), n_simulations=10_000)
    params, info = model.fit(jr.PRNGKey(2), data=data, n_early_stopping_patience=25, n_iter=n_iter)
    inference_result, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)

    plot_posterior(inference_result)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=1_000)
    args = parser.parse_args()
    run(args.n_iter)
