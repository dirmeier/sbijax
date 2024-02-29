import distrax
import tensorflow_probability.substrates.jax as tfp
from jax import random as jr


# ruff: noqa: PLR0913,D417
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
    r"""Sample from a distribution using the No-U-Turn sampler.

    Args:
        rng_key: a jax random key
        lp: the logdensity you wish to sample from
        prior: a function that returns a prior sample
        n_chains: number of chains to sample
        n_samples: number of samples per chain
        n_warmup: number of samples to discard

    Returns:
        a JAX array of dimension n_samples \times n_chains \times len_theta
    """
    init_key, rng_key = jr.split(rng_key)
    initial_states = _slice_init(init_key, n_chains, prior)

    sample_key, rng_key = jr.split(rng_key)
    samples = tfp.mcmc.sample_chain(
        num_results=n_samples - n_warmup,
        current_state=initial_states,
        num_steps_between_results=n_thin,
        kernel=tfp.mcmc.SliceSampler(
            lp, step_size=step_size, max_doublings=n_doubling
        ),
        num_burnin_steps=n_warmup,
        trace_fn=None,
        seed=sample_key,
    )
    return samples


# pylint: disable=missing-function-docstring
def _slice_init(rng_key, n_chains, prior: distrax.Distribution):
    initial_positions = prior(seed=rng_key, sample_shape=(n_chains,))
    return initial_positions
