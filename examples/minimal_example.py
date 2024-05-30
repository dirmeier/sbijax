import distrax
import optax
import jax

from jax import numpy as jnp, random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax import SNL
from sbijax.nn import make_affine_maf
from tensorflow_probability.substrates.jax import distributions as tfd


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        mean=tfd.Normal(jnp.zeros(2), 1.0),
        scale=tfd.HalfNormal(jnp.ones(2)),
    ))
    return prior


def simulator_fn(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["mean"]), 1.0)
    y = theta["mean"] + theta["scale"] * p.sample(seed=seed)
    return y


fns = prior_fn, simulator_fn
model = SNL(fns, make_affine_maf(2))


y_observed = jnp.array([-1.0, 1.0])
data, a = model.simulate_data(jr.PRNGKey(0), n_simulations=10_000)
params, b = model.fit(jr.PRNGKey(1), data=data, optimizer=optax.adam(0.001))
posterior, c = model.sample_posterior(jr.PRNGKey(2), params, y_observed)
