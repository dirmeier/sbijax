import blackjax as bj
import jax
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from jax.flatten_util import ravel_pytree

from sbijax._src.mcmc.sampler import Kernel
from sbijax._src.mcmc.util import burn_in, run_blackjax


# ruff: noqa: PLR0913, D417
def sample_with_imh(
  rng_key, lp, prior, *, n_chains=4, n_samples=2_000, n_warmup=1_000, **kwargs
):
  r"""Draw samples using the independent Metropolis-Hastings sampler.

  Args:
      rng_key: a jax random key
      lp: the logdensity you wish to sample from
      prior: a function that returns a prior sample
      n_chains: number of chains to sample
      n_samples: number of samples per chain returned after the warmup
      n_warmup: number of samples to discard

  Examples:
      >>> import functools as ft
      >>> from jax import numpy as jnp, random as jr
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      ...
      >>> prior = tfd.JointDistributionNamed(
      ...    dict(theta=tfd.Normal(jnp.zeros(2), 1.0))
      ... )
      >>> def log_prob(theta, y):
      ...     lp_prior = prior.log_prob(theta)
      ...     lp_data = tfd.Normal(theta["theta"], 1.0).log_prob(y)
      ...     return jnp.sum(lp_data) + jnp.sum(lp_prior)
      ...
      >>> prop_posterior_lp = ft.partial(log_prob, y=jnp.array([-1.0, 1.0]))
      >>> samples = sample_with_imh(jr.key(0), prop_posterior_lp, prior)

  Returns:
      a tuple ``(samples, info)``: a named pytree with leaves of shape
      ``n_chains x n_samples x dim`` and an
      ``MCMCSampleInfo`` with the mean post-warmup acceptance rate
  """
  init_key, run_key = jr.split(rng_key)
  initial_positions = prior.sample(seed=init_key, sample_shape=(n_chains,))
  return run_blackjax(
    run_key,
    _mh_init,
    initial_positions,
    lp,
    n_chains=n_chains,
    n_samples=n_samples,
    n_warmup=n_warmup,
    **kwargs,
  )


def _irmh_proposal(initial_positions):
  """Build a standard normal proposal over the position and its log-density."""
  position = jax.tree_util.tree_map(lambda x: x[0], initial_positions)
  flat_position, unravel_fn = ravel_pytree(position)

  def proposal_distribution(rng_key):
    return unravel_fn(
      jr.normal(rng_key, flat_position.shape, flat_position.dtype)
    )

  # blackjax evaluates this as the log-density of moving from the first state
  # to the second, so an independent proposal scores the second state
  def proposal_logdensity_fn(_state, other_state):
    flat, _ = ravel_pytree(other_state.position)
    return jnp.sum(jsp.stats.norm.logpdf(flat))

  return proposal_distribution, proposal_logdensity_fn


# pylint: disable=missing-function-docstring,no-member
def _mh_init(rng_key, initial_positions, lp, n_warmup):
  proposal_distribution, proposal_logdensity_fn = _irmh_proposal(
    initial_positions
  )
  kernel = bj.irmh(
    lp,
    proposal_distribution,
    proposal_logdensity_fn=proposal_logdensity_fn,
  )
  step = jax.vmap(kernel.step)
  initial_states = jax.vmap(kernel.init)(initial_positions)
  return burn_in(rng_key, step, initial_states, n_warmup), step


imh = Kernel(init_fn=_mh_init)
