import distrax
import optax
import jax

from jax import numpy as jnp, random as jr
from jax._src.flatten_util import ravel_pytree
from matplotlib import pyplot as plt

from sbijax import SNL
from sbijax.nn import make_affine_maf
from tensorflow_probability.substrates.jax import distributions as tfd


def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        mean=tfd.Normal(jnp.zeros(2), 1.0),
        scale=tfd.HalfNormal(jnp.ones(1)),
    ))
    return prior


def simulator_fn(seed: jr.PRNGKey, theta: dict):
    p = tfd.Normal(jnp.zeros_like(theta["mean"]), 1.0)
    y = theta["mean"] + theta["scale"] * p.sample(seed=seed)
    return y


fns = prior_fn, simulator_fn
model = SNL(fns, make_affine_maf(2))


obs = jnp.array([-1.0, 1.0])
n_rounds = 5

data, params = None, {}
for i in range(n_rounds):
    data, _ = model.simulate_data_and_possibly_append(
        jr.fold_in(jr.PRNGKey(0), i),
        params=params,
        observable=obs,
        data=data,
        sampler="mala",
    )
    params, info = model.fit(jr.fold_in(jr.PRNGKey(1), i), data=data, n_iter=10)
inference_results, diagnostics = model.sample_posterior(
    jr.PRNGKey(2), params, obs
)
print(inference_results)