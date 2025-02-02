"""All-in-one simulation-based inference.

Demonstrates AiO on a linear Gaussian model.
"""
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import plot_posterior
from sbijax.experimental import AiO
from sbijax.experimental.nn import make_simformer_based_score_model


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Normal(jnp.zeros(5), 1)
    ), batch_ndims=0)
    return prior


def simulator_fn(seed, theta):
    mean = theta["theta"].reshape(-1, 5)
    y = tfd.Normal(mean, 0.1).sample(seed=seed)
    return y


def run(n_iter):
    y_observed = jnp.linspace(-2.0, 2.0, 5)
    fns = prior_fn, simulator_fn
    mask = jnp.zeros((10, 10))
    mask = mask.at[np.arange(5, 10), np.arange(5)].set(1)
    mask = mask + mask.T + jnp.eye(10)

    neural_network = make_simformer_based_score_model(5, mask, 1, 1)
    model = AiO(fns, neural_network)

    data, _ = model.simulate_data(jr.PRNGKey(1), n_simulations=10_000)
    params, info = model.fit(jr.PRNGKey(2), data=data, n_early_stopping_patience=25, n_iter=n_iter)
    inference_result, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)

    plot_posterior(inference_result, point_estimate="mean")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=1_000)
    args = parser.parse_args()
    run(args.n_iter)
