"""Simulation-based calibration (SBC).

SBC checks posterior calibration: if a posterior approximation is calibrated,
then for parameters drawn from the prior the rank of each ground-truth
parameter among posterior draws is uniformly distributed. Systematic departures
from uniformity reveal miscalibration (over- or under-dispersed posteriors,
bias). This is the calibration tier of the correctness harness (DR-009).
"""

# ruff: noqa: PLR0913
import jax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.util.data import inference_data_as_dictionary


def sbc(
  rng_key,
  estimator,
  params,
  prior,
  simulator,
  *,
  n_simulations=100,
  n_posterior_samples=1_000,
  **sample_kwargs,
):
  """Compute simulation-based calibration ranks for a fitted estimator.

  For each of ``n_simulations`` draws ``theta* ~ prior`` and
  ``y* ~ simulator(theta*)``, draws ``n_posterior_samples`` from the posterior
  given ``y*`` and ranks each dimension of ``theta*`` among them. Calibrated
  posteriors yield ranks uniform on ``[0, n_posterior_samples]``.

  Args:
      rng_key: a jax random key
      estimator: a fitted :class:`~sbijax._src.inference._estimator.Estimator`
      params: the fitted parameters
      prior: the prior distribution
      simulator: a callable ``(rng_key, theta) -> y``
      n_simulations: number of calibration draws
      n_posterior_samples: posterior draws per calibration draw
      **sample_kwargs: forwarded to ``estimator.sample``

  Returns:
      an integer array of shape ``(n_simulations, n_dims)`` of ranks
  """

  def rank_one(key):
    theta_key, y_key, post_key = jr.split(key, 3)
    theta_true = prior.sample(seed=theta_key, sample_shape=(1,))
    y = simulator(y_key, theta_true)
    idata = estimator.sample(
      post_key,
      params,
      y[0],
      n_samples=n_posterior_samples,
      **sample_kwargs,
    )
    posterior = jax.vmap(lambda x: ravel_pytree(x)[0])(
      inference_data_as_dictionary(idata)
    )
    theta_flat, _ = ravel_pytree(jax.tree.map(lambda a: a[0], theta_true))
    return jnp.sum(posterior < theta_flat, axis=0)

  ranks = [rank_one(k) for k in jr.split(rng_key, n_simulations)]
  return jnp.stack(ranks)
