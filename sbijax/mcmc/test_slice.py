# pylint: skip-file

from functools import partial
from timeit import default_timer as timer

import distrax
import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp

from sbijax.mcmc.slice import sample_with_slice
from sbijax.mcmc.slice_sampler import slice_sampler

p = distrax.Independent(distrax.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0)), 1)


def likelihood_fn(theta, y):
    mu = jnp.tile(theta[:2], 4)
    s1, s2 = theta[2] ** 2, theta[3] ** 2
    corr = s1 * s2 * jnp.tanh(theta[4])
    cov = jnp.array([[s1**2, corr], [corr, s2**2]])
    cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
    p = distrax.MultivariateNormalFullCovariance(mu, cov)
    return p.log_prob(y)


def log_density_fn(theta, y):
    prior_lp = p.log_prob(theta)
    likelihood_lp = likelihood_fn(theta, y)

    lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
    return lp


len_theta = 5
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


def inference_loop(rng_key, kernel, initial_state, n_samples, n_chains):
    @jax.jit
    def _step(states, rng_key):
        keys = jax.random.split(rng_key, n_chains)
        states = jax.vmap(kernel)(keys, states)
        return states, states

    sampling_keys = jax.random.split(rng_key, n_samples)
    _, states = jax.lax.scan(_step, initial_state, sampling_keys)
    return states


#  pylint: disable=too-many-locals,invalid-name,redefined-outer-name
def run_slice(n_samples=30000, n_warmup=10000, n_chains=8, dbl_str="tfp"):
    log_density_partial = partial(log_density_fn, y=y_observed)
    log_density = lambda x: log_density_partial(**x)

    init, kernel = slice_sampler(log_density, 2, dbl_str)
    initial_positions = p.sample(
        seed=random.PRNGKey(1), sample_shape=(n_chains,)
    )
    initial_positions = {"theta": initial_positions}

    initial_states = jax.vmap(init)(initial_positions)
    states = inference_loop(
        random.PRNGKey(23), kernel, initial_states, n_samples, n_chains
    )
    samples = jax.block_until_ready(states.position["theta"])
    samples = samples[n_warmup:, ...].reshape(-1, len_theta)

    return samples


def run_tfp_slice(n_samples=30000, n_warmup=10000, n_chains=8):
    log_density_partial = partial(log_density_fn, y=y_observed)
    log_density = lambda x: jax.vmap(log_density_partial)(x)

    initial_states = p.sample(seed=random.PRNGKey(1), sample_shape=(n_chains,))
    samples = tfp.mcmc.sample_chain(
        num_results=n_samples,
        current_state=initial_states,
        num_steps_between_results=0,
        kernel=tfp.mcmc.SliceSampler(
            log_density, step_size=0.1, max_doublings=2
        ),
        num_burnin_steps=n_warmup,
        trace_fn=None,
        seed=random.PRNGKey(23),
    )

    samples = samples[n_warmup:, ...].reshape(-1, len_theta)
    return samples


start = timer()
s1 = run_slice(dbl_str="custom")
end = timer()
print(f"Custom sampler time: {end - start}")

start = timer()
s2 = run_slice(dbl_str="tfp")
end = timer()
print(f"TFP sampler time: {end - start}")

start = timer()
s3 = run_tfp_slice()
end = timer()
print(f"real TFP sampler time: {end - start}")


fig, axes = plt.subplots(len_theta, 3)
for i in range(len_theta):
    sns.histplot(s1[:, i], color="darkgrey", ax=axes[i, 0])
    sns.histplot(s2[:, i], color="darkgrey", ax=axes[i, 1])
    sns.histplot(s3[:, i], color="darkgrey", ax=axes[i, 2])
    axes[i, 0].set_title(rf"Custom slice")
    axes[i, 1].set_title(rf"Custm slice TFP doubling")
    axes[i, 2].set_title(rf"TFP slice")
    for j in range(3):
        axes[i, j].set_xlim(-5, 5)
sns.despine()
plt.tight_layout()
plt.show()
