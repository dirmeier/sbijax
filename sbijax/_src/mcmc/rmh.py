import blackjax as bj
import jax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.mcmc.sampler import Kernel
from sbijax._src.mcmc.util import run_blackjax


# ruff: noqa: PLR0913, D417
def sample_with_rmh(
  rng_key, lp, prior, *, n_chains=4, n_samples=2_000, n_warmup=1_000, **kwargs
):
  r"""Sample from a distribution using Rosenbluth-Metropolis-Hastings sampler.

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
      >>> samples = sample_with_rmh(jr.key(0), prop_posterior_lp, prior)

  Returns:
      a tuple ``(samples, info)``: a named pytree with leaves of shape
      ``n_chains x (n_samples - n_warmup) x dim`` and an
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
  )


# pylint: disable=missing-function-docstring,no-member
def _mh_init(_rng_key, initial_positions, lp):
  flat_ip = jax.vmap(lambda x: ravel_pytree(x)[0])(initial_positions)
  kernel = bj.rmh(
    lp,
    bj.mcmc.random_walk.normal(jnp.full_like(flat_ip.shape[-1], 0.25)),
  )
  initial_state = jax.vmap(kernel.init)(initial_positions)
  return initial_state, kernel.step


rmh = Kernel(init_fn=_mh_init)
