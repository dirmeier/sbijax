from collections import namedtuple

import arviz as az
import jax


def mcmc_diagnostics(samples: az.InferenceData):
    MCMCDiagnostics = namedtuple("MCMCDiagnostics", "rhat ess")
    return MCMCDiagnostics(az.rhat(samples), az.ess(samples))


# ruff: noqa: PLR0913,D417,E501
def sample_and_post_process_from_blackjax_samples(
    rng_key,
    inf_fn,
    kernel,
    initial_states,
    n_chains,
    n_samples,
    n_warmup,
):
    first_key = list(initial_states.position.keys())[0]
    states = inf_fn(rng_key, kernel, initial_states, n_samples)
    _ = states.position[first_key].block_until_ready()
    thetas = jax.tree_util.tree_map(
        lambda x: x[n_warmup:, ...].reshape(n_chains, n_samples - n_warmup, -1),
        states.position,
    )
    return thetas
