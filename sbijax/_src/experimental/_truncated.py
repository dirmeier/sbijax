"""Truncated-prior proposal for sequential score-based posterior estimation.

Ports the truncated-prior sampling of NPSE/AiO (:cite:t:`sharrock2024sequential`,
:cite:t:`gloeckler2024allinone`) to a standalone proposal compatible with
:func:`~sbijax.run_sequential`'s ``proposal_fn`` hook. A trained posterior
defines a truncation boundary (a low quantile of posterior log-densities) and a
bounding hypercube; the proposal draws uniformly in the hypercube and keeps
draws whose posterior log-density clears the boundary. This resolves the
truncated-proposal open question in the backlog: truncation is a driver option,
not a separate driver.
"""

# ruff: noqa: PLR0913
import jax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.train.sample import sample
from sbijax._src.util.data import flatten_chains


def make_truncated_proposal(
  prior,
  network,
  *,
  quantile=5e-4,
  n_calibration=100_000,
  n_prior=1_000_000,
  max_iter=1_000,
):
  """Build a truncated-prior ``proposal_fn`` for :func:`run_sequential`.

  Args:
      prior: the prior distribution
      network: the score network the estimator wraps (exposes ``log_prob``)
      quantile: lower-tail quantile of posterior log-densities taken as the
          truncation boundary
      n_calibration: number of posterior draws used to calibrate the boundary
          and the bounding hypercube
      n_prior: number of prior draws used to bound the hypercube
      max_iter: maximum rejection rounds before giving up

  Returns:
      a ``proposal_fn(objective, params, observable, sampler)`` returning a
      callable ``(rng_key, n) -> theta`` that draws parameters from the
      truncated prior, in the pytree structure the prior and simulator use
  """
  _, unravel_fn = ravel_pytree(prior.sample(seed=jr.key(0)))

  def proposal_fn(objective, params, observable, sampler=None):
    def log_prob(rng_key, theta_flat):
      return network.apply(
        params,
        rng=rng_key,
        method="log_prob",
        inputs=theta_flat,
        context=jnp.tile(observable, [theta_flat.shape[0], 1]),
        is_training=False,
      )

    def proposal(rng_key, n):
      calib_key, bound_key, rng_key = jr.split(rng_key, 3)
      samples, _ = sample(
        calib_key,
        objective,
        params,
        observable,
        sampler=sampler,
        n_samples=n_calibration,
      )
      flat_posterior = jax.vmap(lambda x: ravel_pytree(x)[0])(
        flatten_chains(samples)
      )
      lp_key, rng_key = jr.split(rng_key)
      boundary = jnp.quantile(log_prob(lp_key, flat_posterior), quantile)

      flat_prior = jax.vmap(lambda x: ravel_pytree(x)[0])(
        prior.sample(seed=bound_key, sample_shape=(n_prior,))
      )
      lo = jnp.maximum(flat_posterior.min(0), flat_prior.min(0))
      hi = jnp.minimum(flat_posterior.max(0), flat_prior.max(0))

      accepted, n_curr, it = [], 0, 0
      while n_curr < n and it < max_iter:
        u_key, lp_key, rng_key = jr.split(rng_key, 3)
        cand = jr.uniform(u_key, (n, lo.shape[0]), minval=lo, maxval=hi)
        keep = cand[log_prob(lp_key, cand) > boundary]
        accepted.append(keep)
        n_curr += keep.shape[0]
        it += 1
      if it == max_iter:
        raise ValueError("truncated proposal did not converge")
      thetas = jnp.concatenate(accepted, axis=0)[:n]
      return jax.vmap(unravel_fn)(thetas)

    return proposal

  return proposal_fn
