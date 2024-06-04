import distrax
import optax
import jax

from jax import numpy as jnp, random as jr
from jax._src.flatten_util import ravel_pytree
from matplotlib import pyplot as plt

from sbijax import SNL
from sbijax.nn import make_affine_maf
from tensorflow_probability.substrates.jax import distributions as tfd


from functools import partial
from jax import scipy as jsp
from sbijax.mcmc import sample_with_nuts, sample_with_slice
#
# def prior_fn():
#     prior = tfd.JointDistributionNamed(dict(
#         mean=tfd.Normal(jnp.zeros(2), 1.0),
#         scale=tfd.HalfNormal(jnp.ones(1)),
#     ))
#     return prior
#
#
# def simulator_fn(seed: jr.PRNGKey, theta: dict):
#     p = tfd.Normal(jnp.zeros_like(theta["mean"]), 1.0)
#     y = theta["mean"] + theta["scale"] * p.sample(seed=seed)
#     return y
#
#
# fns = prior_fn, simulator_fn
# model = SNL(fns, make_affine_maf(2))
#
#
# obs = jnp.array([-1.0, 1.0])
# n_rounds = 5
#
# data, params = None, {}
# for i in range(n_rounds):
#     data, _ = model.simulate_data_and_possibly_append(
#         jr.fold_in(jr.PRNGKey(0), i),
#         params=params,
#         observable=obs,
#         data=data,
#         sampler="mala",
#     )
#     params, info = model.fit(jr.fold_in(jr.PRNGKey(1), i), data=data, n_iter=10)
# inference_results, diagnostics = model.sample_posterior(
#     jr.PRNGKey(2), params, obs
# )
# print(inference_results)
def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))
    ))
    return prior


def simulator_fn(seed, theta):
    theta = theta["theta"]
    orig_shape = theta.shape
    if theta.ndim == 2:
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
    us = tfd.Normal(0.0, 1.0).sample(
        seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
    )
    xs = jnp.empty_like(us)
    xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
    y = xs.at[:, :, :, 1].set(
        s1 * (r * us[:, :, :, 0] + jnp.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
    )
    if len(orig_shape) == 2:
        y = y.reshape((*theta.shape[:1], 8))
    else:
        y = y.reshape((*theta.shape[:2], 8))
    return y


def likelihood_fn(theta, y):
    mu = jnp.tile(theta[:2], 4)
    s1, s2 = theta[2] ** 2, theta[3] ** 2
    corr = s1 * s2 * jnp.tanh(theta[4])
    cov = jnp.array([[s1**2, corr], [corr, s2**2]])
    cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
    p = tfd.MultivariateNormalFullCovariance(mu, cov)
    return p.log_prob(y)


def log_density_fn(theta, y):
    prior_lp = tfd.JointDistributionNamed(dict(
        theta=tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))
    )).log_prob(theta)
    likelihood_lp = likelihood_fn(theta, y)
    lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
    return lp

true_theta = prior_fn().sample(seed=jr.PRNGKey(12345), sample_shape=(1,))
obs = simulator_fn(jr.PRNGKey(123456), true_theta)


log_density = partial(log_density_fn, y=obs)

def lp(theta):
    return jax.vmap(log_density)(theta)

slice_samples = sample_with_slice(
    jr.PRNGKey(0),
    lp,
    prior_fn().sample,
    n_chains=10,
    n_samples=20_000,
    n_warmup=10_000
)