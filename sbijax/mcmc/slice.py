import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from sbijax.mcmc.slice_sampler import slice_sampler


# pylint: disable=too-many-arguments
def sample_with_slice(
    rng_seq,
    lp,
    n_chains,
    n_samples,
    n_warmup,
    prior,
    n_thin=0,
    n_doubling=5,
    implementation="custom",
    **kwargs,
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

    if implementation == "custom":
        fn = _custom_slice
    else:
        fn = _tfp_slice
    return fn(
        rng_seq,
        lp,
        n_chains,
        n_samples,
        n_warmup,
        prior,
        n_thin,
        n_doubling,
        **kwargs,
    )


# pylint: disable=too-many-arguments
def _tfp_slice(
    rng_seq,
    lp,
    n_chains,
    n_samples,
    n_warmup,
    prior,
    n_thin=0,
    n_doubling=5,
    step_size=1.0,
):
    initial_states = prior(seed=next(rng_seq), sample_shape=(n_chains,))
    samples = tfp.mcmc.sample_chain(
        num_results=n_samples,
        current_state=initial_states,
        num_steps_between_results=n_thin,
        kernel=tfp.mcmc.SliceSampler(
            lp, step_size=step_size, max_doublings=n_doubling
        ),
        num_burnin_steps=n_warmup,
        trace_fn=None,
        seed=next(rng_seq),
    )
    return samples


# pylint: disable=too-many-arguments
def _custom_slice(
    rng_seq,
    lp,
    n_chains,
    n_samples,
    n_warmup,
    prior,
    n_thin=0,
    n_doubling=5,
    **kwargs,
):
    def _inference_loop(rng_key, kernel, initial_state, n_samples):
        @jax.jit
        def _step(states, rng_key):
            keys = jax.random.split(rng_key, n_chains)
            states = jax.vmap(kernel)(keys, states)
            return states, states

        sampling_keys = jax.random.split(rng_key, n_samples)
        _, states = jax.lax.scan(_step, initial_state, sampling_keys)
        return states

    initial_states, kernel = _slice_init(
        rng_seq, prior, n_chains, lp, n_doubling
    )

    n_total_samples = _minimal_sample_size_with_thinning(
        n_samples, n_warmup, n_thin
    )
    states = _inference_loop(
        next(rng_seq), kernel, initial_states, n_total_samples
    )
    _ = states.position["theta"].block_until_ready()
    thetas = states.position["theta"][n_warmup:, :, :]
    # thinning: take the n_thin-th sample as first point and then step to
    # the next sample by skipping n_thin indexes, i.e.
    thetas = thetas[n_thin :: (n_thin + 1), ...]
    return thetas


# pylint: disable=missing-function-docstring
def _slice_init(rng_seq, prior, n_chains, lp, n_doublings):
    initial_positions = prior(seed=next(rng_seq), sample_shape=(n_chains,))
    initial_positions = {"theta": initial_positions}
    init, kernel = slice_sampler(lp, n_doublings)
    initial_states = jax.vmap(init)(initial_positions)
    return initial_states, kernel


def _minimal_sample_size_with_thinning(n_samples, n_warmup, n_thin):
    n_effective_samples = n_samples - n_warmup
    n_to_draw = n_effective_samples
    while True:
        n_returned = len(np.arange(n_to_draw)[n_thin :: (n_thin + 1)])
        if n_returned >= n_effective_samples:
            break
        n_to_draw += 1
    return n_warmup + n_to_draw
