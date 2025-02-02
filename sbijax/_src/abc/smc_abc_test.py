# pylint: skip-file

import jax
from jax import numpy as jnp
from jax import random as jr

from sbijax import SMCABC


def distance_fn(y_simulated, y_observed):
    diff = y_simulated - y_observed
    dist = jax.vmap(lambda el: jnp.linalg.norm(el))(diff)
    return dist


def test_smcabc(prior_simulator_tuple):
    y_observed = jnp.array([-1.0, 1.0])
    estim = SMCABC(prior_simulator_tuple, lambda x: x, distance_fn)
    estim.sample_posterior(jr.PRNGKey(0), y_observed, n_rounds=1, n_particles=1_000)
