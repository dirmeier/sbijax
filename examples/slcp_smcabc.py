"""
SLCP example from [1] using SMCABC
"""

from functools import partial

import distrax
import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp

from sbijax import SMCABC
from sbijax.mcmc import sample_with_nuts


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
    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    smc = SMCABC(fns, summary_fn, distance_fn)
    smc.fit(23, y_observed)
    smc_samples = smc.sample_posterior(5, 1000, 10, 0.9, 500)

    def log_density_fn(theta, y):
        prior_lp = prior_logdensity_fn(theta)
        likelihood_lp = likelihood_fn(theta, y)

        lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
        return lp

    log_density_partial = partial(log_density_fn, y=y_observed)
    log_density = lambda x: log_density_partial(**x)

    rng_seq = hk.PRNGSequence(12)
    nuts_samples = sample_with_nuts(
        rng_seq, log_density, len_theta, 4, 20000, 5000
    )
    nuts_samples = nuts_samples.reshape(-1, len_theta)

    g = sns.PairGrid(pd.DataFrame(nuts_samples))
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
        sns.histplot(nuts_samples[:, i], color="darkgrey", ax=axes[i, 0])
        sns.histplot(smc_samples[:, i], color="darkblue", ax=axes[i, 1])
        axes[i, 0].set_title(rf"Sampled posterior $\theta_{i}$")
        axes[i, 1].set_title(rf"Approximated posterior $\theta_{i}$")
        for j in range(2):
            axes[i, j].set_xlim(-5, 5)
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
