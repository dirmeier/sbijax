"""
Example using consistency model posterior estimation on a bivariate Gaussian
"""
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax.mcmc import sample_with_slice


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Normal(jnp.full(2, 0.0), 1.0)
    ), batch_ndims=0)
    return prior


def _map_fun_inv(theta, y):
    ang = jnp.array([-jnp.pi / 4.0])
    c = jnp.cos(ang)
    s = jnp.sin(ang)
    z0 = (c * theta[:, 0] - s * theta[:, 1]).reshape(-1, 1)
    z1 = (s * theta[:, 0] + c * theta[:, 1]).reshape(-1, 1)
    return y - jnp.concatenate([-jnp.abs(z0), z1], axis=1)


def likelihood_fn(y, theta):
    theta = theta.reshape(1, 2)
    p = _map_fun_inv(theta, y).reshape(1, -1)

    u = p[:, 0] - 0.25
    v = p[:, 1]

    r = jnp.sqrt(u ** 2 + v ** 2)
    ll = -0.5 * ((r - 0.1) / 0.01) ** 2 - 0.5 * jnp.log(2 * jnp.array([jnp.pi]) * 0.01 ** 2)
    ll = jnp.where(
        u < 0.0,
        jnp.array(-jnp.inf),
        ll
    )
    return ll


def log_density_fn(theta, y):
    prior_lp = prior_fn().log_prob(theta)
    likelihood_lp = likelihood_fn(y, theta)
    lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
    #jax.debug.print("🤯 {x} 🤯", x=lp)
    return lp


def run():
    y_observed = jnp.array([-0.6396706, 0.16234657])

    log_density_partial = partial(log_density_fn, y=y_observed)
    log_density = lambda x: jax.vmap(log_density_partial)(x)
    samples = sample_with_slice(
        jr.PRNGKey(0),
        log_density,
        prior_fn().sample, n_chains=4, n_samples=10000, n_warmup=5000
    )
    samples = np.array(samples.reshape(-1, 2))
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()


if __name__ == "__main__":
    run()
