import blackjax as bj
import jax
from jax import random as jr

from sbijax._src.mcmc.sampler import Kernel
from sbijax._src.mcmc.util import run_blackjax


def _nuts_init(rng_key, initial_positions, lp):
  n_chains = jax.tree_util.tree_leaves(initial_positions)[0].shape[0]
  init_keys = jr.split(rng_key, n_chains)
  warmup = bj.window_adaptation(bj.nuts, lp)
  initial_states, kernel_params = jax.vmap(
    lambda seed, param: warmup.run(seed, param)[0]
  )(init_keys, initial_positions)
  kernel_params = {k: v[0] for k, v in kernel_params.items()}
  _, kernel = bj.nuts(lp, **kernel_params)
  return initial_states, kernel


nuts = Kernel(init_fn=_nuts_init)


# ruff: noqa: PLR0913, D417
def sample_with_nuts(
  rng_key, lp, prior, *, n_chains=4, n_samples=2_000, n_warmup=1_000, **kwargs
):
  r"""Sample from a distribution using the No-U-Turn sampler.

  Args:
      rng_key: a jax random key
      lp: the logdensity you wish to sample from
      prior: a function that returns a prior sample
      n_chains: number of chains to sample
      n_samples: number of samples per chain
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
      >>> samples = sample_with_nuts(jr.key(0), prop_posterior_lp, prior)

  Returns:
      a tuple ``(samples, info)``: a named pytree with leaves of shape
      ``n_chains x (n_samples - n_warmup) x dim`` and an
      ``MCMCSampleInfo`` with the mean post-warmup acceptance rate
  """
  init_key, run_key = jr.split(rng_key)
  initial_positions = prior.sample(seed=init_key, sample_shape=(n_chains,))
  return run_blackjax(
    run_key,
    _nuts_init,
    initial_positions,
    lp,
    n_chains=n_chains,
    n_samples=n_samples,
    n_warmup=n_warmup,
  )
