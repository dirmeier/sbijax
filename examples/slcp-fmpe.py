"""Flow matching posterior estimation.

Demonstrates FMPE on the simple likelihood complex posterior model.
"""
import optax
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import FMPE
from sbijax import inference_data_as_dictionary
from sbijax.nn import make_cnf


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))
    ), batch_ndims=0)
    return prior


def simulator_fn(seed, theta):
    theta = theta["theta"]
    theta = theta[:, None, :]
    us_key, noise_key = jr.split(seed)

    def _unpack_params(ps):
        m0 = ps[..., [0]]
        m1 = ps[..., [1]]
        s0 = ps[..., [2]] ** 2
        s1 = ps[..., [3]] ** 2
        r = jnp.tanh(ps[..., [4]])
        return m0, m1, s0, s1, r

    m0, m1, s0, s1, r = _unpack_params(theta)
    us = tfd.Normal(0.0, 1.0).sample(
        seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
    )
    xs = jnp.empty_like(us)
    xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
    y = xs.at[:, :, :, 1].set(
        s1 * (r * us[:, :, :, 0] + jnp.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
    )
    y = y.reshape((*theta.shape[:1], 8))
    return y


def run(n_iter):
    y_observed = jnp.array([[
        -0.9707123,
        -2.9461224,
        -0.4494722,
        -3.4231849,
        -0.13285634,
        -3.364017,
        -0.85367596,
        -2.4271638,
    ]])

    n_dim_theta = 5
    n_layers, hidden_size = 5, 128
    neural_network = make_cnf(n_dim_theta, n_layers, hidden_size)
    fns = prior_fn, simulator_fn
    fmpe = FMPE(fns, neural_network)

    data, _ = fmpe.simulate_data(
        jr.PRNGKey(1),
        n_simulations=20_000,
    )
    fmpe_params, info = fmpe.fit(
        jr.PRNGKey(2),
        data=data,
        optimizer=optax.adam(0.001),
        n_iter=n_iter,
        n_early_stopping_delta=0.00001,
        n_early_stopping_patience=30
    )
    inference_results, diagnostics = fmpe.sample_posterior(
        jr.PRNGKey(5), fmpe_params, y_observed, n_samples=25_000
    )

    samples = inference_data_as_dictionary(inference_results.posterior)["theta"]
    _, axes = plt.subplots(figsize=(12, 10), nrows=5, ncols=5)
    for i in range(0, 5):
        for j in range(0, 5):
            ax = axes[i, j]
            if i < j:
                ax.axis('off')
            else:
                ax.hexbin(samples[..., j], samples[..., i], gridsize=50, bins='log')
    for i in range(5):
        axes[i, i].hist(samples[..., i], color="black")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=1_000)
    args = parser.parse_args()
    run(args.n_iter)
