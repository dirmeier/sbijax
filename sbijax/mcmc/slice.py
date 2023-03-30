import distrax
import tensorflow_probability.substrates.jax as tfp


# pylint: disable=too-many-arguments
def sample_with_slice(
    rng_seq,
    lp,
    n_chains,
    n_samples,
    n_warmup,
    prior,
    n_thin=2,
    n_doubling=5,
    step_size=1,
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

    initial_states = _slice_init(rng_seq, n_chains, prior)
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


# pylint: disable=missing-function-docstring
def _slice_init(rng_seq, n_chains, prior: distrax.Distribution):
    initial_positions = prior(seed=next(rng_seq), sample_shape=(n_chains,))

    return initial_positions
