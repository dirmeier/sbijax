"""Surjective neural likelihood estimation example.

Demonstrates sequential surjective neural likelihood estimation on the simple
 likelihood complex posterior model.
"""
import argparse

import distrax
import haiku as hk
import jax
import matplotlib.pyplot as plt
import optax
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from surjectors import (
    AffineMaskedAutoregressiveInferenceFunnel,
    Chain,
    MaskedAutoregressive,
    Permutation,
    TransformedDistribution,
)
from surjectors.nn import MADE, make_mlp
from surjectors.util import unstack
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import SNLE, inference_data_as_dictionary
from sbijax.nn import make_maf


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
    us = distrax.Normal(0.0, 1.0).sample(
        seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
    )
    xs = jnp.empty_like(us)
    xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
    y = xs.at[:, :, :, 1].set(
        s1 * (r * us[:, :, :, 0] + jnp.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
    )
    y = y.reshape((*theta.shape[:1], 8))
    return y


def likelihood_fn(theta, y):
    mu = jnp.tile(theta[:2], 4)
    s1, s2 = theta[2] ** 2, theta[3] ** 2
    corr = s1 * s2 * jnp.tanh(theta[4])
    cov = jnp.array([[s1**2, corr], [corr, s2**2]])
    cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
    p = distrax.MultivariateNormalFullCovariance(mu, cov)
    return p.log_prob(y)


def log_density_fn(theta, y):
    prior_lp = distrax.Independent(
        distrax.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0)), 1
    ).log_prob(theta)
    likelihood_lp = likelihood_fn(theta, y)

    lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
    return lp


def make_model(dim, use_surjectors):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _decoder_fn(n_dim):
        decoder_net = make_mlp(
            [50, n_dim * 2],
            w_init=hk.initializers.TruncatedNormal(stddev=0.001),
        )

        def _fn(z):
            params = decoder_net(z)
            mu, log_scale = jnp.split(params, 2, -1)
            return distrax.Independent(
                distrax.Normal(mu, jnp.exp(log_scale)), 1
            )

        return _fn

    def _flow(method, **kwargs):
        layers = []
        n_dimension = dim
        order = jnp.arange(n_dimension)
        for i in range(5):
            if i == 2 and use_surjectors:
                n_latent = 6
                layer = AffineMaskedAutoregressiveInferenceFunnel(
                    n_latent,
                    _decoder_fn(n_dimension - n_latent),
                    conditioner=MADE(
                        n_latent,
                        [64, 64],
                        2,
                        w_init=hk.initializers.TruncatedNormal(0.001),
                        b_init=jnp.zeros,
                        activation=jax.nn.tanh,
                    ),
                )
                n_dimension = n_latent
                order = order[::-1]
                order = order[:n_dimension] - jnp.min(order[:n_dimension])
            else:
                layer = MaskedAutoregressive(
                    bijector_fn=_bijector_fn,
                    conditioner=MADE(
                        n_dimension,
                        [64, 64],
                        2,
                        w_init=hk.initializers.TruncatedNormal(0.001),
                        b_init=jnp.zeros,
                        activation=jax.nn.tanh,
                    ),
                )
                order = order[::-1]
            layers.append(layer)
            layers.append(Permutation(order, 1))
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dimension), jnp.ones(n_dimension)),
            reinterpreted_batch_ndims=1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


def run(n_rounds, n_iter):
    y_obs = jnp.array([[
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

    neural_network = make_maf(8, n_layer_dimensions=[8, 8, 5, 5, 5])
    snl = SNLE(fns, neural_network)
    optimizer = optax.adam(1e-3)

    data, params = None, {}
    for i in range(n_rounds):
        data, _ = snl.simulate_data_and_possibly_append(
            jr.fold_in(jr.PRNGKey(1), i),
            params=params,
            observable=y_obs,
            data=data,
        )
        params, info = snl.fit(
            jr.fold_in(jr.PRNGKey(2), i), data=data, optimizer=optimizer, n_iter=n_iter
        )

    sample_key, rng_key = jr.split(jr.PRNGKey(3))
    inference_results, _ = snl.sample_posterior(sample_key, params, y_obs)

    samples = inference_data_as_dictionary(inference_results.posterior)["theta"]
    _, axes = plt.subplots(figsize=(12, 10), nrows=5, ncols=5)
    for i in range(0, 5):
        for j in range(0, 5):
            ax = axes[i, j]
            if i < j:
                ax.axis('off')
            else:
                ax.hexbin(samples[..., j], samples[..., i], gridsize=50,
                          bins='log')
    for i in range(5):
        axes[i, i].hist(samples[..., i], color="black")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=15)
    parser.add_argument("--n-iter", type=int, default=1_000)
    args = parser.parse_args()
    run(args.n_rounds, args.n_iter)
