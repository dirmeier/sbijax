"""Generic training driver over an objective's TrainFns (design B)."""

import logging

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from sbijax._src.train._types import Info, next_round
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


# ruff: noqa: PLR0913
def fit(
  rng_key,
  objective,
  data,
  *,
  optimizer=optax.adam(3e-4),
  info=None,
  n_iter=1000,
  batch_size=100,
  percentage_data_as_validation_set=0.1,
  n_early_stopping_patience=10,
  n_early_stopping_delta=1e-3,
):
  """Train any objective's ``TrainFns`` with early stopping.

  Args:
      rng_key: a jax random key
      objective: an ``ObjectiveFns``/``SummaryFns`` (anything with ``train``)
      data: a ``{"y", "theta"}`` dataset pytree
      optimizer: an optax optimizer, bound into the primitives here
      info: previous round's ``Info`` (``None`` for round 0)
      n_iter: number of epochs
      batch_size: minibatch size
      percentage_data_as_validation_set: validation split fraction
      n_early_stopping_patience: early-stopping patience
      n_early_stopping_delta: minimum early-stopping improvement

  Returns:
      a tuple ``(params, Info)``
  """
  train_fns = objective.train
  itr_key, rng_key = jr.split(rng_key)
  train_iter, val_iter = as_batch_iterators(
    itr_key, data, batch_size, 1.0 - percentage_data_as_validation_set, True
  )
  init_key, rng_key = jr.split(rng_key)
  state = train_fns.init_fn(optimizer, init_key, next(iter(train_iter)))

  step_fn = jax.jit(lambda rng, s, b: train_fns.step_fn(optimizer, rng, s, b))
  eval_fn = jax.jit(train_fns.eval_fn)

  def _weighted(metric_fn, itr):
    total = 0.0
    for batch in itr:
      total += metric_fn(batch) * (batch["y"].shape[0] / itr.num_samples)
    return total

  losses = np.zeros([n_iter, 2])
  early_stop = EarlyStopping(n_early_stopping_delta, n_early_stopping_patience)
  best_params, best_loss = state.params, np.inf
  i = 0
  for i in tqdm(range(n_iter)):
    rng_key = jr.fold_in(rng_key, i)
    train_loss = 0.0
    for batch in train_iter:
      step_key, rng_key = jr.split(rng_key)
      metrics, state = step_fn(step_key, state, batch)
      train_loss += metrics["loss"] * (batch["y"].shape[0] / train_iter.num_samples)
    val_key, rng_key = jr.split(rng_key)
    val_loss = _weighted(
      lambda b, s=state, k=val_key: eval_fn(k, s, b)["loss"], val_iter
    )
    losses[i] = jnp.array([train_loss, val_loss])
    _, early_stop = early_stop.update(val_loss)
    if early_stop.should_stop:
      break
    if val_loss < best_loss:
      best_loss, best_params = val_loss, state.params

  stacked = jnp.vstack(losses)[: (i + 1), :]
  return best_params, Info(round=next_round(info), losses=stacked)
