from collections import namedtuple

import arviz as az
import jax
import xarray
from jax import random as jr


def mcmc_diagnostics(samples: xarray.DataTree):
  MCMCDiagnostics = namedtuple("MCMCDiagnostics", "rhat ess")
  return MCMCDiagnostics(az.rhat(samples), az.ess(samples))


def _inference_loop(rng_key, kernel, initial_state, n_chains, n_samples):
  @jax.jit
  def _step(states, rng_key):
    keys = jr.split(rng_key, n_chains)
    states, _ = jax.vmap(kernel)(keys, states)
    return states, states

  sampling_keys = jr.split(rng_key, n_samples)
  _, states = jax.lax.scan(_step, initial_state, sampling_keys)
  return states


# ruff: noqa: PLR0913
def run_blackjax(rng_key, init_fn, prior, lp, *, n_chains, n_samples, n_warmup):
  """Draw samples from a distribution using a BlackJAX kernel.

  Constructs the initial chain states and kernel via ``init_fn``, runs a
  vectorised (over chains) sampling loop, discards the warmup draws and
  reshapes the result to ``n_chains x (n_samples - n_warmup) x dim``.

  Args:
      rng_key: a jax random key
      init_fn: a callable ``(rng_key, n_chains, prior, lp) ->
          (initial_states, kernel_step)`` constructing the initial BlackJAX
          chain states and the kernel step function
      prior: a distribution to sample the initial chain positions from
      lp: the logdensity to sample from
      n_chains: number of chains to sample
      n_samples: number of samples per chain
      n_warmup: number of samples to discard

  Returns:
      a JAX pytree with keys corresponding to the variable names and tensor
      values of dimension ``n_chains x (n_samples - n_warmup) x dim_variable``
  """
  init_key, sample_key = jr.split(rng_key)
  initial_states, kernel = init_fn(init_key, n_chains, prior, lp)
  first_key = list(initial_states.position.keys())[0]
  states = _inference_loop(
    sample_key, kernel, initial_states, n_chains, n_samples
  )
  _ = states.position[first_key].block_until_ready()
  thetas = jax.tree_util.tree_map(
    lambda x: x[n_warmup:, ...].reshape(n_chains, n_samples - n_warmup, -1),
    states.position,
  )
  return thetas
