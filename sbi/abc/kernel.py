from jax import numpy as jnp


def gaussian_kernel(x, h):
    d = jnp.sum(jnp.square(x / h) ** 2, axis=2)
    k = jnp.exp(-0.5 * d) / jnp.sqrt(2.0 * jnp.pi)
    return k / h
