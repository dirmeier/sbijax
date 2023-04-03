"""
SLCP example from [1] using SNL and masked coupling bijections or surjections
"""

import argparse
from functools import partial

import distrax
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp
from jax import vmap
from surjectors import Chain, TransformedDistribution
from surjectors.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors.bijectors.permutation import Permutation
from surjectors.conditioners import mlp_conditioner
from surjectors.nn.made import MADE
from surjectors.surjectors.affine_masked_autoregressive_inference_funnel import (  # type: ignore # noqa: E501
    AffineMaskedAutoregressiveInferenceFunnel,
)
from surjectors.util import unstack

from sbijax import SNL
from sbijax.mcmc.slice import sample_with_slice


def prior_model_fns():
    p = distrax.Independent(
        distrax.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0)), 1
    )
    return p.sample, p.log_prob


def likelihood_fn(theta, y):
    mu = jnp.tile(theta[:2], 4)
    s1, s2 = theta[2] ** 2, theta[3] ** 2
    corr = s1 * s2 * jnp.tanh(theta[4])
    cov = jnp.array([[s1**2, corr], [corr, s2**2]])
    cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
    p = distrax.MultivariateNormalFullCovariance(mu, cov)
    return p.log_prob(y)


def simulator_fn(seed, theta):
    orig_shape = theta.shape
    if theta.ndim == 2:
        theta = theta[:, None, :]
    us_key, noise_key = random.split(seed)

    def _unpack_params(ps):
        m0 = ps[..., [0]]
        m1 = ps[..., [1]]
        s0 = ps[..., [2]] ** 2
        s1 = ps[..., [3]] ** 2
        r = np.tanh(ps[..., [4]])
        return m0, m1, s0, s1, r

    m0, m1, s0, s1, r = _unpack_params(theta)
    us = distrax.Normal(0.0, 1.0).sample(
        seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
    )
    xs = jnp.empty_like(us)
    xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
    y = xs.at[:, :, :, 1].set(
        s1 * (r * us[:, :, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
    )
    if len(orig_shape) == 2:
        y = y.reshape((*theta.shape[:1], 8))
    else:
        y = y.reshape((*theta.shape[:2], 8))
    return y


def make_model(dim, use_surjectors):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _decoder_fn(n_dim):
        decoder_net = mlp_conditioner([32, 32, n_dim * 2])

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
                n_latent = 7
                layer = AffineMaskedAutoregressiveInferenceFunnel(
                    n_latent,
                    _decoder_fn(n_dimension - n_latent),
                    MADE(n_latent, [32, 32, n_latent * 2], 2),
                )
                n_dimension = n_latent
                order = order[::-1]
                order = order[:n_dimension] - jnp.min(order[:n_dimension])
            else:
                layer = MaskedAutoregressive(
                    bijector_fn=_bijector_fn,
                    conditioner=MADE(n_dimension, [32, 32, n_dimension * 2], 2),
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


def run(use_surjectors):
    len_theta = 5
    # this is the thetas used in SNL
    # thetas = jnp.array([-0.7, -2.9, -1.0, -0.9, 0.6])
    y_observed = jnp.array(
        [
            [
                -0.9707123,
                -2.9461224,
                -0.4494722,
                -3.4231849,
                -0.13285634,
                -3.364017,
                -0.85367596,
                -2.4271638,
            ]
        ]
    )
    prior_sampler, prior_fn = prior_model_fns()
    fns = (prior_sampler, prior_fn), simulator_fn
    model = make_model(y_observed.shape[1], use_surjectors)
    snl = SNL(fns, model)
    optimizer = optax.adam(1e-3)
    params, info = snl.fit(
        random.PRNGKey(23), y_observed, optimizer, n_rounds=3, sampler="slice"
    )

    snl_samples, _ = snl.sample_posterior(
        params, 4, 20000, 10000, sampler="slice"
    )
    snl_samples = snl_samples.reshape(-1, len_theta)

    def log_density_fn(theta, y):
        prior_lp = prior_fn(theta)
        likelihood_lp = likelihood_fn(theta, y)

        lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
        return lp

    log_density_partial = partial(log_density_fn, y=y_observed)
    log_density = lambda x: vmap(log_density_partial)(x)

    rng_seq = hk.PRNGSequence(12)
    slice_samples = sample_with_slice(
        rng_seq, log_density, 4, 20000, 10000, prior_sampler
    )
    slice_samples = slice_samples.reshape(-1, len_theta)

    g = sns.PairGrid(pd.DataFrame(slice_samples))
    g.map_upper(sns.scatterplot, color="black", marker=".", edgecolor=None, s=2)
    g.map_diag(plt.hist, color="black")
    for ax in g.axes.flatten():
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    g.fig.set_figheight(5)
    g.fig.set_figwidth(5)
    plt.show()

    fig, axes = plt.subplots(len_theta, 2)
    for i in range(len_theta):
        sns.histplot(slice_samples[:, i], color="darkgrey", ax=axes[i, 0])
        sns.histplot(snl_samples[:, i], color="darkblue", ax=axes[i, 1])
        axes[i, 0].set_title(rf"Sampled posterior $\theta_{i}$")
        axes[i, 1].set_title(rf"Approximated posterior $\theta_{i}$")
        for j in range(2):
            axes[i, j].set_xlim(-5, 5)
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-surjectors", action="store_true", default=True)
    args = parser.parse_args()
    run(args.use_surjectors)
