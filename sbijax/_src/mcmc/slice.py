import jax
import tensorflow_probability.substrates.jax as tfp
from einops import rearrange
from jax import random as jr
from jax._src.flatten_util import ravel_pytree


# ruff: noqa: PLR0913,D417,E501
def sample_with_slice(
    rng_key,
    lp,
    prior,
    *,
    n_chains=4,
    n_samples=2_000,
    n_warmup=1_000,
    n_thin=2,
    n_doubling=5,
    step_size=1,
    **kwargs,
):
    r"""Sample from a distribution using a slice sampler.

    Args:
        rng_key: a jax random key
        lp: the logdensity you wish to sample from
        prior: a function that returns a prior sample
        n_chains: number of chains to sample
        n_samples: number of samples per chain
        n_warmup: number of samples to discard
        n_thin: integer specifying how many samples to discard between
            draws
        n_doubling: maximum number of doubling steps
        step_size: floating number specifying the size of each step

    Examples:
        >>> import functools as ft
        >>> from jax import numpy as jnp, random as jr
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = tfd.JointDistributionNamed(
        ...     dict(
        ...         mean=tfd.Normal(jnp.zeros(2), 1.0),
        ...         std=tfd.HalfNormal(1.0)
        ...     ),
        ...     batch_ndims=0,
        ... )
        >>> def log_prob(theta, y):
        ...     lp_prior = prior.log_prob(theta)
        ...     lp_data = tfd.Normal(theta["mean"], theta["std"]).log_prob(y)
        ...     return jnp.sum(lp_data) + jnp.sum(lp_prior)
        ...
        >>> prop_posterior_lp = ft.partial(log_prob, y=jnp.array([-1.0, 1.0]))
        >>> samples = sample_with_slice(jr.PRNGKey(0), prop_posterior_lp, prior)

    Returns:
        a JAX pytree with keys corresponding to the variables names
        and tensor values of dimension `n_chains x n_samples x dim_variable`
    """
    test_sample = prior.sample(seed=jr.PRNGKey(0))
    _, unravel_fn = ravel_pytree(test_sample)

    def lp__(theta):
        return jax.vmap(lambda x: lp(unravel_fn(x)))(theta)

    init_key, rng_key = jr.split(rng_key)
    initial_states = _slice_init(init_key, n_chains, prior)
    initial_states = jax.vmap(lambda x: ravel_pytree(x)[0])(initial_states)

    sample_key, rng_key = jr.split(rng_key)
    samples = tfp.mcmc.sample_chain(
        num_results=n_samples - n_warmup,
        current_state=initial_states,
        num_steps_between_results=n_thin,
        kernel=tfp.mcmc.SliceSampler(
            lp__, step_size=step_size, max_doublings=n_doubling
        ),
        num_burnin_steps=n_warmup,
        trace_fn=None,
        seed=sample_key,
    )
    samples = rearrange(samples, "s c v -> (s c) v")
    samples = jax.vmap(unravel_fn)(samples)
    samples = {
        k: v.reshape(n_chains, (n_samples - n_warmup), -1)
        for k, v in samples.items()
    }
    return samples


# pylint: disable=missing-function-docstring
def _slice_init(rng_key, n_chains, prior):
    initial_positions = prior.sample(seed=rng_key, sample_shape=(n_chains,))
    return initial_positions
