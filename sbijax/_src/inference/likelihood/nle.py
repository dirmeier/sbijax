"""Neural likelihood estimation.

Implements the method introduced in :cite:t:`papama2019neural` as a functional
estimator: a factory that composes a prior, a conditional density network, and
an MCMC sampler into an :class:`~sbijax._src.inference._estimator.Estimator`.
"""

# ruff: noqa: PLR0913
from typing import NamedTuple

import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.inference._estimator import Estimator, next_round
from sbijax._src.mcmc.nuts import sample_with_nuts
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.train import train_loop


class NLEInfo(NamedTuple):
  """Diagnostics returned by :func:`nle`'s ``fit`` (also SNLE; DR-011).

  Attributes:
      round: the training round; ``fit`` reads this back to advance rounds
      losses: a ``(n_epochs, 2)`` array of train/validation losses
  """

  round: int
  losses: jax.Array


def nle(prior, network, *, sampler=sample_with_nuts):
  """Construct a neural likelihood estimator.

  The network models the likelihood ``p(y | theta)``; the posterior is obtained
  by combining it with the prior and drawing samples with ``sampler``.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      network: a conditional density estimator exposing a ``log_prob`` method
      sampler: an MCMC sampler ``(rng_key, lp, prior, **kwargs) -> samples``
          used to draw from the posterior; defaults to NUTS

  Returns:
      an :class:`~sbijax._src.inference._estimator.Estimator`
  """

  def fit(
    rng_key,
    data,
    *,
    info=None,
    optimizer=None,
    n_iter=1000,
    batch_size=100,
    percentage_data_as_validation_set=0.1,
    n_early_stopping_patience=10,
  ):
    if optimizer is None:
      optimizer = optax.adam(0.0003)
    itr_key, rng_key = jr.split(rng_key)
    train_iter, val_iter = as_batch_iterators(
      itr_key,
      data,
      batch_size,
      1.0 - percentage_data_as_validation_set,
      True,
    )
    init_key, rng_key = jr.split(rng_key)
    init_batch = next(iter(train_iter))
    params = network.init(
      init_key,
      method="log_prob",
      y=init_batch["y"],
      x=init_batch["theta"],
    )

    def loss_fn(params, rng, **batch):  # noqa: ARG001
      lp = network.apply(
        params,
        rng=None,
        method="log_prob",
        y=batch["y"],
        x=batch["theta"],
      )
      return -jnp.mean(lp)

    params, losses = train_loop(
      rng_key,
      params=params,
      optimizer=optimizer,
      loss_fn=loss_fn,
      validation_loss_fn=loss_fn,
      train_iter=train_iter,
      val_iter=val_iter,
      n_iter=n_iter,
      n_early_stopping_patience=n_early_stopping_patience,
    )
    return params, NLEInfo(round=next_round(info), losses=losses)

  def sample(
    rng_key,
    params,
    observable,
    *,
    n_chains=4,
    n_samples=2_000,
    n_warmup=1_000,
    **kwargs,
  ):
    """Draw posterior samples via MCMC.

    Returns:
        a tuple ``(samples, info)`` of the named posterior pytree and an
        ``MCMCSampleInfo``
    """
    observable = jnp.atleast_2d(observable)

    def log_density(theta):
      theta_flat, _ = ravel_pytree(theta)
      theta_tiled = jnp.tile(theta_flat, [observable.shape[0], 1])
      log_lik = network.apply(
        params,
        rng=None,
        method="log_prob",
        y=observable,
        x=theta_tiled,
      )
      return jnp.sum(log_lik) + jnp.sum(prior.log_prob(theta))

    samples, info = sampler(
      rng_key=rng_key,
      lp=log_density,
      prior=prior,
      n_chains=n_chains,
      n_samples=n_samples,
      n_warmup=n_warmup,
      **kwargs,
    )
    return samples, info

  return Estimator(fit=fit, sample=sample)
