import jax

from sbijax.mcmc._slice import slice_sampler


# pylint: disable=too-many-arguments
def sample_with_slice(
    rng_seq,
    lp,
    n_chains,
    n_samples,
    n_warmup,
    prior,
    n_thin=1,
    n_doubling=5,
    step_size=1
):
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
            states = jax.vmap(kernel)(keys, states)
            return states, states

        sampling_keys = jax.random.split(rng_key, n_samples)
        _, states = jax.lax.scan(_step, initial_state, sampling_keys)
        return states

    initial_states, kernel = _slice_init(rng_seq, prior, n_chains, lp, n_doubling)
    states = _inference_loop(next(rng_seq), kernel, initial_states, n_samples)
    _ = states.position["theta"].block_until_ready()
    thetas = states.position["theta"][n_warmup:, :, :]
    # thinning: take the n_thin-th sample as first point and then step to
    # the next sample by skipping n_thin indexes, i.e.
    thetas = thetas[n_thin::(n_thin + 1), ...]
    return thetas



# pylint: disable=missing-function-docstring
def _slice_init(rng_seq, prior, n_chains, lp, n_doublings):
    initial_positions = prior(seed=next(rng_seq), sample_shape=(n_chains,))
    initial_positions = {"theta": initial_positions}

    init, kernel = slice_sampler(lp, n_doublings)
    initial_states = jax.vmap(init)(initial_positions)

    return initial_states, kernel
