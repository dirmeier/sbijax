import blackjax as bj
import distrax
import jax
from jax import random as jr


# pylint: disable=too-many-arguments,unused-argument
def sample_with_nuts(
    rng_key, lp, prior, *, n_chains=4, n_samples=2_000, n_warmup=1_000, **kwargs
):
    r"""Sample from a distribution using the No-U-Turn sampler.

    Args:
        rng_key: hk.PRNGSequence
            a hk.PRNGSequence
        lp: Callable
            the logdensity you wish to sample from
        prior: Callable
            a function that returns a prior sample
        n_chains: int
            number of chains to sample
        n_samples: int
            number of samples per chain
        n_warmup: int
            number of samples to discard

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
    initial_states, kernel = _nuts_init(init_key, n_chains, prior, lp)

    states = _inference_loop(init_key, kernel, initial_states, n_samples)
    _ = states.position["theta"].block_until_ready()
    thetas = states.position["theta"][n_warmup:, :, :]

    return thetas


# pylint: disable=missing-function-docstring
def _nuts_init(rng_key, n_chains, prior: distrax.Distribution, lp):
    init_key, rng_key = jr.split(rng_key)
    initial_positions = prior(seed=init_key, sample_shape=(n_chains,))
    initial_positions = {"theta": initial_positions}

    init_keys = jr.split(rng_key, n_chains)
    warmup = bj.window_adaptation(bj.nuts, lp)
    initial_states, kernel_params = jax.vmap(
        lambda seed, param: warmup.run(seed, param)[0]
    )(init_keys, initial_positions)

    kernel_params = {k: v[0] for k, v in kernel_params.items()}
    _, kernel = bj.nuts(lp, **kernel_params)

    return initial_states, kernel
