from typing import Callable

import distrax
import jax

from sbijax.mcmc import _slice_sampler


# pylint: disable=too-many-arguments
def sample_with_slice(rng_seq, lp, n_chains, n_samples, n_warmup, prior):
    """
    Sample from a distribution using the No-U-Turn sampler.

    Parameters
    ----------
    rng_seq: hk.PRNGSequence
        a hk.PRNGSequence
    lp: Callable
        the logdensity you wish to sample from
    n_chains: int
        number of chains to sample
    n_samples: int
        number of samples per chain
    n_warmup: int
        number of samples to discard

    Returns
    -------
    jnp.ndarrau
        a JAX array of dimension n_samples \times n_chains \times len_theta
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

    initial_states, kernel = _slice_init(rng_seq, n_chains, lp, prior)
    states = _inference_loop(next(rng_seq), kernel, initial_states, n_samples)
    _ = states.position["theta"].block_until_ready()
    thetas = states.position["theta"][n_warmup:, :, :]
    return thetas


# pylint: disable=missing-function-docstring
def _slice_init(
    rng_seq, n_chains, logdensity_fn: Callable, prior: distrax.Distribution
):
    slice = _slice_sampler.slice(logdensity_fn, prior)
    initial_positions = prior.sample(
        seed=next(rng_seq), sample_shape=(n_chains,)
    )
    initial_positions = {"theta": initial_positions}
    initial_states = jax.vmap(slice.init, in_axes=(0))(initial_positions)
    kernel = slice.step

    return initial_states, kernel
