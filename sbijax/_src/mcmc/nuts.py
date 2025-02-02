import blackjax as bj
import jax
from jax import random as jr

from sbijax._src.mcmc.util import sample_and_post_process_from_blackjax_samples


# ruff: noqa: PLR0913,D417,E501
def sample_with_nuts(
    rng_key, lp, prior, *, n_chains=4, n_samples=2_000, n_warmup=1_000, **kwargs
):
    r"""Sample from a distribution using the No-U-Turn sampler.

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
        >>> samples = sample_with_nuts(jr.PRNGKey(0), prop_posterior_lp, prior)

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
    initial_states, kernel = _nuts_init(init_key, n_chains, prior, lp)
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


# pylint: disable=missing-function-docstring
def _nuts_init(rng_key, n_chains, prior, lp):
    init_key, rng_key = jr.split(rng_key)
    initial_positions = prior.sample(seed=init_key, sample_shape=(n_chains,))

    init_keys = jr.split(rng_key, n_chains)
    warmup = bj.window_adaptation(bj.nuts, lp)
    initial_states, kernel_params = jax.vmap(
        lambda seed, param: warmup.run(seed, param)[0]
    )(init_keys, initial_positions)

    kernel_params = {k: v[0] for k, v in kernel_params.items()}
    _, kernel = bj.nuts(lp, **kernel_params)

    return initial_states, kernel
