import jax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.diagnostics.convergence import mcmc_convergence
from sbijax._src.inference._sample_info import MCMCSampleInfo


def _inference_loop(rng_key, kernel, initial_state, n_chains, n_samples):
  @jax.jit
  def _step(states, rng_key):
    keys = jr.split(rng_key, n_chains)
    states, infos = kernel(keys, states)
    return states, (states, infos)

  sampling_keys = jr.split(rng_key, n_samples)
  _, (states, infos) = jax.lax.scan(_step, initial_state, sampling_keys)
  return states, infos


def burn_in(rng_key, kernel, states, n_warmup):
  """Advance every chain by ``n_warmup`` steps and return the final states.

  Args:
      rng_key: a jax random key
      kernel: ``(keys, states) -> (states, infos)`` advancing every chain
      states: the chain states, with a leading chain axis
      n_warmup: number of steps

  Returns:
      the chain states after ``n_warmup`` steps
  """
  n_chains = jax.tree_util.tree_leaves(states.position)[0].shape[0]

  def _step(states, rng_key):
    states, _ = kernel(jr.split(rng_key, n_chains), states)
    return states, None

  states, _ = jax.lax.scan(_step, states, jr.split(rng_key, n_warmup))
  return states


# ruff: noqa: PLR0913
def run_blackjax(
  rng_key,
  init_fn,
  initial_positions,
  lp,
  *,
  n_chains,
  n_samples,
  n_warmup,
  **kernel_kwargs,
):
  """Draw samples from a distribution using a BlackJAX kernel.

  Args:
      rng_key: a jax random key
      init_fn: ``(rng_key, initial_positions, lp, n_warmup, **kernel_kwargs)
          -> (states, kernel)`` where ``states`` are the chain states after
          the warmup and ``kernel(keys, states)`` advances every chain by one
          step
      initial_positions: a named pytree of chain start positions with a leading
          ``n_chains`` axis on every leaf
      lp: the logdensity to sample from
      n_chains: number of chains
      n_samples: number of samples per chain returned after the warmup
      n_warmup: number of warmup steps, run by ``init_fn``
      **kernel_kwargs: forwarded to ``init_fn``

  Returns:
      ``(samples, MCMCSampleInfo)`` — see module docs.
  """
  init_key, sample_key = jr.split(rng_key)
  initial_states, kernel = init_fn(
    init_key, initial_positions, lp, n_warmup, **kernel_kwargs
  )
  first_key = list(initial_states.position.keys())[0]
  states, infos = _inference_loop(
    sample_key, kernel, initial_states, n_chains, n_samples
  )
  _ = states.position[first_key].block_until_ready()
  thetas = jax.tree_util.tree_map(
    lambda x: jnp.swapaxes(x, 0, 1).reshape(n_chains, n_samples, -1),
    states.position,
  )
  acceptance = jnp.mean(infos.acceptance_rate)
  rhat, ess = mcmc_convergence(thetas, n_chains)
  return thetas, MCMCSampleInfo(acceptance_rate=acceptance, rhat=rhat, ess=ess)
