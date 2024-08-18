"""NASS+SMCABC example.

Demonstrates neural approximate sufficient statistics with SMCABC on the
simple likelihood complex posterior model.
"""
import argparse

import distrax
import jax
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import NASS, SMCABC, inference_data_as_dictionary
from sbijax.nn import make_nass_net


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))
    ), batch_ndims=0)
    return prior


def simulator_fn(seed, theta):
    theta = theta["theta"]
    orig_shape = theta.shape
    if theta.ndim == 2:
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
    us = distrax.Normal(0.0, 1.0).sample(
        seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
    )
    xs = jnp.empty_like(us)
    xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
    y = xs.at[:, :, :, 1].set(
        s1 * (r * us[:, :, :, 0] + jnp.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
    )
    if len(orig_shape) == 2:
        y = y.reshape((*theta.shape[:1], 8))
    else:
        y = y.reshape((*theta.shape[:2], 8))
    return y


def distance_fn(y_simulated, y_observed):
    diff = y_simulated - y_observed
    dist = jax.vmap(lambda el: jnp.linalg.norm(el))(diff)
    return dist


def run(n_rounds):
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
    fns = prior_fn, simulator_fn
    model_nass = NASS(fns, make_nass_net([64, 64, 5], [64, 64, 1]))

    data, _ = model_nass.simulate_data(jr.PRNGKey(1), n_simulations=20_000)
    params_nass, _ = model_nass.fit(jr.PRNGKey(2), data=data, n_early_stopping_patience=25)

    def summary_fn(y):
        s = model_nass.summarize(params_nass, y)
        return s

    model_smc = SMCABC(fns, summary_fn, distance_fn)
    inference_results, _ = model_smc.sample_posterior(
        jr.PRNGKey(3), y_observed, n_rounds=n_rounds, n_particles=5_000, eps_step=0.825, ess_min=2_000
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=15)
    args = parser.parse_args()
    run(args.n_rounds)
