"""Model from Lueckmann et al., 2021, https://arxiv.org/abs/2101.04653.

Adopted from https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/two_moons/task.py.
"""

import jax
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

simulator_params = {
    "a_low": -jnp.pi / 2.0,
    "a_high": jnp.pi / 2.0,
    "base_offset": 0.25,
    "r_loc": 0.1,
    "r_scale": 0.01,
}

ang = jnp.array([-jnp.pi / 4.0])
cos_ang, sin_ang = jnp.cos(ang), jnp.sin(ang)


def _map_fun_inv(theta, y):
    z0 = cos_ang * theta[:, [0]] - sin_ang * theta[:, [1]]
    z1 = sin_ang * theta[:, [0]] + cos_ang * theta[:, [1]]
    return y - jnp.concatenate([-jnp.abs(z0), z1], axis=1)


def _map_fun_(theta, p):
    z0 = cos_ang * theta[:, [0]] - sin_ang * theta[:, [1]]
    z1 = sin_ang * theta[:, [0]] + cos_ang * theta[:, [1]]
    return p + jnp.concatenate([-jnp.abs(z0), z1], axis=1)


def two_moons():
    def prior_fn():
        prior = tfd.JointDistributionNamed(
            dict(
                theta=tfd.Independent(
                    tfd.Uniform(
                        -jnp.ones(2),
                        jnp.ones(2),
                    ),
                    reinterpreted_batch_ndims=1,
                )
            )
        )
        return prior

    def simulator(seed, theta):
        theta = theta["theta"].reshape(-1, 2)
        n = theta.shape[0]

        a_key, r_key = jr.split(seed)
        a_dist = tfd.Uniform(
            simulator_params["a_low"], simulator_params["a_high"]
        )
        a = a_dist.sample(seed=a_key, sample_shape=(n, 1))
        r_dist = tfd.Normal(
            simulator_params["r_loc"], simulator_params["r_scale"]
        )
        r = r_dist.sample(seed=r_key, sample_shape=(n, 1))

        p = jnp.concatenate(
            [
                jnp.cos(a) * r + simulator_params["base_offset"],
                jnp.sin(a) * r,
            ],
            axis=1,
        )

        ret = _map_fun_(theta.reshape(-1, 2), p)
        return ret

    def likelihood(y, theta):
        def fn(y, theta):
            theta, y = theta.reshape(-1, 2), y.reshape(-1, 2)
            p = _map_fun_inv(theta, y).reshape(1, -1)
            u = p[:, [0]] - simulator_params["base_offset"]
            v = p[:, [1]]

            r = jnp.sqrt(u**2 + v**2)
            ll = (
                -0.5
                * (
                    (r - simulator_params["r_loc"])
                    / simulator_params["r_scale"]
                )
                ** 2
            )
            ll = ll - 0.5 * jnp.log(
                2 * jnp.pi * simulator_params["r_scale"] ** 2
            )

            # I think this (isntead if -inf) is necessary for slice sampling
            ll = jnp.where(u < 0.0, -100_000.0, ll)
            return jnp.squeeze(ll)

        theta, y = theta["theta"].reshape(-1, 2), y.reshape(-1, 2)
        log_lik = jax.vmap(fn)(y, theta)
        return log_lik

    return prior_fn(), simulator, likelihood
