"""Flow matching posterior estimation.

Implements the FMPE algorithm of :cite:t:`wilderberger2023flow` as a functional
estimator. The network is a continuous normalizing flow modelling the posterior
directly, so ``sample`` draws from it and rejects draws outside the prior
support; no MCMC sampler is involved.
"""

# ruff: noqa: PLR0913
from typing import NamedTuple

import jax
import optax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.inference._estimator import Estimator, next_round
from sbijax._src.inference.posterior._sampling import rejection_sample_flow
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.train import train_loop


class FMPEInfo(NamedTuple):
  """Diagnostics returned by :func:`fmpe`'s ``fit`` (DR-011).

  Attributes:
      round: the training round; ``fit`` reads this back to advance rounds
      losses: a ``(n_epochs, 2)`` array of train/validation losses
  """

  round: int
  losses: jax.Array


def fmpe(prior, network):
  """Construct a flow matching posterior estimator.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      network: a continuous normalizing flow with ``loss`` and ``sample``
          methods

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
    n_early_stopping_delta=1e-3,
  ):
    if optimizer is None:
      optimizer = optax.adam(0.0003)
    itr_key, rng_key = jr.split(rng_key)
    train_iter, val_iter = as_batch_iterators(
      itr_key, data, batch_size, 1.0 - percentage_data_as_validation_set, True
    )
    init_key, rng_key = jr.split(rng_key)
    init_batch = next(iter(train_iter))
    params = network.init(
      init_key,
      method="loss",
      inputs=init_batch["theta"],
      context=init_batch["y"],
      is_training=False,
    )

    def loss_fn(params, rng, **batch):
      lp = network.apply(
        params,
        rng=rng,
        method="loss",
        inputs=batch["theta"],
        context=batch["y"],
        is_training=True,
      )
      return jnp.mean(lp)

    def validation_loss_fn(params, rng, **batch):
      lp = network.apply(
        params,
        rng=rng,
        method="loss",
        inputs=batch["theta"],
        context=batch["y"],
        is_training=False,
      )
      return jnp.mean(lp)

    params, losses = train_loop(
      rng_key,
      params=params,
      optimizer=optimizer,
      loss_fn=loss_fn,
      validation_loss_fn=validation_loss_fn,
      train_iter=train_iter,
      val_iter=val_iter,
      n_iter=n_iter,
      n_early_stopping_patience=n_early_stopping_patience,
      n_early_stopping_delta=n_early_stopping_delta,
    )
    return params, FMPEInfo(round=next_round(info), losses=losses)

  def sample(rng_key, params, observable, *, n_samples=4_000, **kwargs):
    return rejection_sample_flow(
      rng_key, network, params, prior, observable, n_samples
    )

  return Estimator(fit=fit, sample=sample)
