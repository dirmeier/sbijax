import jax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.diagnostics.convergence import mcmc_convergence
from sbijax._src.inference._sample_info import MCMCSampleInfo


def _inference_loop(rng_key, kernel, initial_state, n_chains, n_samples):
  @jax.jit
  def _step(states, rng_key):
    keys = jr.split(rng_key, n_chains)
    states, infos = jax.vmap(kernel)(keys, states)
    return states, (states, infos)

  sampling_keys = jr.split(rng_key, n_samples)
  _, (states, infos) = jax.lax.scan(_step, initial_state, sampling_keys)
  return states, infos


# ruff: noqa: PLR0913
def run_blackjax(rng_key, init_fn, prior, lp, *, n_chains, n_samples, n_warmup):
  """Draw samples from a distribution using a BlackJAX kernel.

  Runs a vectorised (over chains) sampling loop on the named-pytree position,
  discards the warmup draws, and returns the named-pytree samples together with
  a mean post-warmup acceptance rate.

  Args:
      rng_key: a jax random key
      init_fn: a callable ``(rng_key, n_chains, prior, lp) -> (initial_states,
          kernel_step)`` constructing the initial BlackJAX chain states and the
          kernel step function
      prior: a distribution to sample the initial chain positions from
      lp: the logdensity to sample from
      n_chains: number of chains to sample
      n_samples: number of samples per chain (including warmup)
      n_warmup: number of leading samples to discard

  Returns:
      a tuple ``(samples, info)`` where ``samples`` is the named pytree with
      leaves of shape ``n_chains x (n_samples - n_warmup) x dim`` and ``info``
      is an :class:`~sbijax._src.inference._sample_info.MCMCSampleInfo`
  """
  init_key, sample_key = jr.split(rng_key)
  initial_states, kernel = init_fn(init_key, n_chains, prior, lp)
  first_key = list(initial_states.position.keys())[0]
  states, infos = _inference_loop(
    sample_key, kernel, initial_states, n_chains, n_samples
  )
  _ = states.position[first_key].block_until_ready()
  thetas = jax.tree_util.tree_map(
    lambda x: x[n_warmup:, ...].reshape(n_chains, n_samples - n_warmup, -1),
    states.position,
  )
  acceptance = jnp.mean(infos.acceptance_rate[n_warmup:, ...])
  rhat, ess = mcmc_convergence(thetas, n_chains)
  return thetas, MCMCSampleInfo(acceptance_rate=acceptance, rhat=rhat, ess=ess)
