import blackjax as bj
import distrax
import jax
from jax import numpy as jnp
from jax import random


# pylint: disable=too-many-arguments
def sample_with_nuts(rng_seq, lp, len_theta, n_chains, n_samples, n_warmup):
    """
    Sample from a distribution using the No-U-Turn sampler.

    Parameters
    ----------
    rng_seq: hk.PRNGSequence
        a hk.PRNGSequence
    lp: Callable
        the logdensity you wish to sample from
    len_theta: int
        the number of parameters to sample
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

    initial_states, kernel = _nuts_init(rng_seq, len_theta, n_chains, lp)
    states = _inference_loop(next(rng_seq), kernel, initial_states, n_samples)
    _ = states.position["theta"].block_until_ready()
    thetas = states.position["theta"][n_warmup:, :, :]
    return thetas


# pylint: disable=missing-function-docstring
def _nuts_init(rng_seq, len_theta, n_chains, lp):
    initial_positions = distrax.MultivariateNormalDiag(
        jnp.zeros(len_theta),
        jnp.ones(len_theta),
    ).sample(seed=next(rng_seq), sample_shape=(n_chains,))
    initial_positions = {"theta": initial_positions}

    init_keys = random.split(next(rng_seq), n_chains)
    warmup = bj.window_adaptation(bj.nuts, lp)
    initial_states, kernel_params = jax.vmap(
        lambda seed, param: warmup.run(seed, param)[0]
    )(init_keys, initial_positions)

    kernel_params = {k: v[0] for k, v in kernel_params.items()}
    _, kernel = bj.nuts(lp, **kernel_params)

    return initial_states, kernel
