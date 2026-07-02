"""The uniform interface for learned summary statistics."""

# ruff: noqa: PLR0913
from collections.abc import Callable
from typing import NamedTuple

import jax
import optax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.util.dataloader import (
  as_batch_iterators,
  as_numpy_iterator_from_slices,
)
from sbijax._src.util.train import train_loop


class SummaryInfo(NamedTuple):
  """Diagnostics returned by a :class:`SummaryNet`'s ``fit`` (DR-011).

  Summary networks are not sequential, so -- unlike an estimator ``Info`` --
  this record carries no ``round`` field, only the loss history.

  Attributes:
      losses: a ``(n_epochs, 2)`` array of train/validation losses
  """

  losses: jax.Array


class SummaryNet(NamedTuple):
  """A learned summary-statistics network.

  Summary methods learn a low-dimensional statistic of the data rather than a
  posterior. ``fit`` trains the network and ``summarize`` maps data through it;
  the resulting summaries are typically fed to an
  :class:`~sbijax._src.inference._estimator.Estimator`.

  Attributes:
      fit: ``(rng_key, data, **kwargs) -> (params, SummaryInfo)``
      summarize: ``(params, data, **kwargs) -> summaries``
  """

  fit: Callable
  summarize: Callable


def make_summary_net(network, jsd_loss):
  """Build a :class:`SummaryNet` from a network and a JSD summary loss.

  Args:
      network: a summary network with ``forward`` and ``summary`` methods
      jsd_loss: a callable ``(params, rng, apply_fn, **batch) -> scalar``

  Returns:
      a :class:`SummaryNet`
  """

  def fit(
    rng_key,
    data,
    *,
    optimizer=None,
    n_iter=1000,
    batch_size=128,
    percentage_data_as_validation_set=0.1,
    n_early_stopping_patience=10,
  ):
    if optimizer is None:
      optimizer = optax.adam(0.0003)
    itr_key, rng_key = jr.split(rng_key)
    train_iter, val_iter = as_batch_iterators(
      itr_key, data, batch_size, 1.0 - percentage_data_as_validation_set, True
    )
    init_key, rng_key = jr.split(rng_key)
    init_batch = next(iter(train_iter))
    params = network.init(init_key, method="forward", **init_batch)

    def loss_fn(params, rng, **batch):
      return jsd_loss(params, rng, network.apply, **batch)

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
    return params, SummaryInfo(losses=losses)

  def summarize(params, data, *, batch_size=512):
    if params is None or len(params) == 0:
      return data
    y = {"y": data} if isinstance(data, jnp.ndarray) else data
    itr = as_numpy_iterator_from_slices(y, batch_size)

    @jax.jit
    def _summarize(batch):
      return network.apply(params, method="summary", y=batch["y"])

    summaries = jnp.concatenate([_summarize(batch) for batch in itr], axis=0)
    if isinstance(data, dict):
      ret = data.copy()
      ret["y"] = summaries
      return ret
    return summaries

  return SummaryNet(fit=fit, summarize=summarize)
