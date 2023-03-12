"""
SLCP example from [1] using SNL and masked coupling bijections or surjections
"""

import distrax
import jax
import matplotlib.pyplot as plt
import seaborn as sns
from jax import numpy as jnp

from sbijax import RejectionABC


def prior_model_fns(leng):
    p = distrax.Independent(
        distrax.Uniform(jnp.full(leng, -2.0), jnp.full(leng, 2.0)), 1
    )
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.MultivariateNormalDiag(theta, 0.1 * jnp.ones_like(theta))
    y = p.sample(seed=seed)
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


def kernel_fn(y_simulated, y_observed, scale=1.0):
    diffs = y_simulated - y_observed

    def _kern(diff):
        d = jnp.square(jnp.linalg.norm(diff / scale))
        k = jnp.exp(-0.5 * d) / jnp.sqrt(2.0 * jnp.pi)
        return k

    ks = jax.vmap(_kern)(diffs)
    return ks / scale


def run():
    print(kernel_fn(jnp.zeros((1, 2, 2)), jnp.zeros((1, 2, 2))))
    len_thetas = 2
    y_observed = jnp.array([-1.0, 1.0])

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns(len_thetas)
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn

    snl = RejectionABC(fns, summary_fn, kernel_fn)
    snl.fit(23, y_observed)
    snl_samples = snl.sample_posterior(1000, 10000, 3.0, 0.1)

    fig, axes = plt.subplots(len_thetas)
    for i in range(len_thetas):
        sns.histplot(snl_samples[:, i], color="darkblue", ax=axes[i])
        axes[i].set_title(rf"Approximated posterior $\theta_{i}$")
    sns.despine()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
