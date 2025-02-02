import blackjax as bj
import jax
from jax import random as jr

from sbijax._src.mcmc.util import sample_and_post_process_from_blackjax_samples


# ruff: noqa: PLR0913,D417,E501
def sample_with_imh(
    rng_key, lp, prior, *, n_chains=4, n_samples=2_000, n_warmup=1_000, **kwargs
):
    r"""Draw samples using the independent Metropolis-Hastings sampler.

    Args:
        rng_key: a jax random key
        lp: the logdensity you wish to sample from
        prior: a function that returns a prior sample
        n_chains: number of chains to sample
        n_samples: number of samples per chain
        n_warmup: number of samples to discard

    Examples:
        >>> import functools as ft
        >>> from jax import numpy as jnp, random as jr
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = tfd.JointDistributionNamed(
        ...    dict(theta=tfd.Normal(jnp.zeros(2), 1.0))
        ... )
        >>> def log_prob(theta, y):
        ...     lp_prior = prior.log_prob(theta)
        ...     lp_data = tfd.Normal(theta["theta"], 1.0).log_prob(y)
        ...     return jnp.sum(lp_data) + jnp.sum(lp_prior)
        ...
        >>> prop_posterior_lp = ft.partial(log_prob, y=jnp.array([-1.0, 1.0]))
        >>> samples = sample_with_imh(jr.PRNGKey(0), prop_posterior_lp, prior)

    Returns:
        a JAX pytree with keys corresponding to the variables names
        and tensor values of dimension `n_chains x n_samples x dim_variable`
    """

    def _inference_loop(rng_key, kernel, initial_state, n_samples):
        @jax.jit
        def _step(states, rng_key):
            keys = jax.random.split(rng_key, n_chains)
            states, _ = jax.vmap(kernel)(keys, states)
            return states, states

        sampling_keys = jax.random.split(rng_key, n_samples)
        _, states = jax.lax.scan(_step, initial_state, sampling_keys)
        return states

    init_key, rng_key = jr.split(rng_key)
    initial_states, kernel = _mh_init(init_key, n_chains, prior, lp)

    samples = sample_and_post_process_from_blackjax_samples(
        rng_key,
        _inference_loop,
        kernel,
        initial_states,
        n_chains,
        n_samples,
        n_warmup,
    )
    return samples


# ruff: noqa: E731
def _irmh_proposal_distribution(initial_positions):
    shape = lambda p: () if p.ndim == 1 else (p.shape[-1],)

    def fn(rng_key):
        return {
            k: jax.random.normal(rng_key, shape=shape(v))
            for k, v in initial_positions.items()
        }

    return fn


# pylint: disable=missing-function-docstring,no-member
def _mh_init(rng_key, n_chains, prior, lp):
    init_key, rng_key = jr.split(rng_key)
    initial_positions = prior.sample(seed=init_key, sample_shape=(n_chains,))
    kernel = bj.irmh(lp, _irmh_proposal_distribution(initial_positions))
    initial_state = jax.vmap(kernel.init)(initial_positions)
    return initial_state, kernel.step
