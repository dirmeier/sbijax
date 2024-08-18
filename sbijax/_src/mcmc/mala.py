import blackjax as bj
import jax
from jax import random as jr


# ruff: noqa: PLR0913,D417
def sample_with_mala(
    rng_key, lp, prior, *, n_chains=4, n_samples=2_000, n_warmup=1_000, **kwargs
):
    r"""Sample from a distribution using the MALA sampler.

    Args:
        rng_key: a hk.PRNGSequence
        lp: the logdensity you wish to sample from
        prior: a function that returns a prior sample
        n_chains: number of chains to sample
        n_samples: number of samples per chain
        n_warmup: number of samples to discard

    Returns:
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

    init_key, rng_key = jr.split(rng_key)
    initial_states, kernel = _mala_init(init_key, n_chains, prior, lp)

    first_key = list(initial_states.position.keys())[0]
    states = _inference_loop(init_key, kernel, initial_states, n_samples)
    _ = states.position[first_key].block_until_ready()
    theta = jax.tree_util.tree_map(
        lambda x: x[n_warmup:, ...].reshape(n_chains, n_samples - n_warmup, -1),
        states.position,
    )
    return theta


# pylint: disable=missing-function-docstring,no-member
def _mala_init(rng_key, n_chains, prior, lp):
    init_key, rng_key = jr.split(rng_key)
    initial_positions = prior(seed=init_key, sample_shape=(n_chains,))

    kernel = bj.mala(lp, 0.1)
    initial_state = jax.vmap(kernel.init)(initial_positions)
    return initial_state, kernel.step
