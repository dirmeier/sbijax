from collections import namedtuple

import blackjax as bj
import distrax
import jax
from jax import numpy as jnp
from jax import random


def sample_with_nuts(rng_seq, lp, len_theta, n_chains, n_samples, n_warmup):
    def _inference_loop(rng_key, kernel, initial_state, n_samples):
        @jax.jit
        def _step(states, rng_key):
            keys = jax.random.split(rng_key, n_chains)
            states, infos = jax.vmap(kernel)(keys, states)
            return states, states

        sampling_keys = jax.random.split(rng_key, n_samples)
        _, states = jax.lax.scan(_step, initial_state, sampling_keys)
        return states

    initial_states, kernel = _nuts_init(rng_seq, len_theta, n_chains, lp)
    states = _inference_loop(next(rng_seq), kernel, initial_states, n_samples)
    _ = states.position["theta"].block_until_ready()
    thetas = states.position["theta"][n_warmup:, :, :]

    return thetas


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


def mcmc_diagnostics(samples):
    n_theta = samples.shape[-1]
    esses = [0] * n_theta
    rhats = [0] * n_theta
    for i in range(n_theta):
        posterior = samples[:, :, i].T
        esses[i] = bj.diagnostics.effective_sample_size(posterior)
        rhats[i] = bj.diagnostics.potential_scale_reduction(posterior)
    return namedtuple("diagnostics", "ess rhat")(esses, rhats)
