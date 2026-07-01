"""Shared training loop for neural estimators."""

import logging

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from sbijax._src.util.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


def _weighted_mean(per_batch_fn, itr):
  """Sum a per-batch quantity, weighting each batch by its sample fraction.

  Args:
      per_batch_fn: a callable mapping a batch to a scalar
      itr: a data iterator exposing ``num_samples``

  Returns:
      the sample-size weighted mean over all batches
  """
  total = 0.0
  for batch in itr:
    total += per_batch_fn(batch) * (batch["y"].shape[0] / itr.num_samples)
  return total


def train_loop(  # noqa: PLR0913
  rng_key,
  *,
  params,
  optimizer,
  loss_fn,
  validation_loss_fn,
  train_iter,
  val_iter,
  n_iter,
  n_early_stopping_patience,
  n_early_stopping_delta=1e-3,
):
  """Train a model by minimising a loss with early stopping.

  Iterates ``n_iter`` epochs, accumulating a sample-weighted training loss
  over ``train_iter`` and evaluating a sample-weighted validation loss over
  ``val_iter``. Keeps the parameters that achieved the lowest validation loss
  and stops early when the validation loss stops improving.

  The optimizer is owned by the loop: it calls ``optimizer.init`` and performs
  ``value_and_grad`` / ``update`` / ``apply_updates`` internally, so callers
  only supply per-batch loss functions.

  Args:
      rng_key: a jax random key
      params: the initial (already constructed) model parameters
      optimizer: an optax optimizer
      loss_fn: a callable ``(params, rng, **batch) -> scalar`` training loss
      validation_loss_fn: a callable ``(params, rng, **batch) -> scalar``
          validation loss
      train_iter: training data iterator exposing ``num_samples``
      val_iter: validation data iterator exposing ``num_samples``
      n_iter: number of epochs
      n_early_stopping_patience: patience of the early stopping criterion
      n_early_stopping_delta: minimum improvement of the early stopping
          criterion

  Returns:
      a tuple of the best parameters and an array of stacked
      ``(train_loss, validation_loss)`` values per epoch
  """
  state = optimizer.init(params)

  @jax.jit
  def step(params, rng, state, **batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, rng, **batch)
    updates, new_state = optimizer.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_state

  losses = np.zeros([n_iter, 2])
  early_stop = EarlyStopping(n_early_stopping_delta, n_early_stopping_patience)
  best_params, best_loss = None, np.inf
  logger.info("training model")
  i = 0
  for i in tqdm(range(n_iter)):
    train_loss = 0.0
    rng_key = jr.fold_in(rng_key, i)
    for batch in train_iter:
      train_key, rng_key = jr.split(rng_key)
      batch_loss, params, state = step(params, train_key, state, **batch)
      train_loss += batch_loss * (batch["y"].shape[0] / train_iter.num_samples)
    val_key, rng_key = jr.split(rng_key)
    validation_loss = _weighted_mean(
      lambda batch, params=params, val_key=val_key: validation_loss_fn(
        params, val_key, **batch
      ),
      val_iter,
    )
    losses[i] = jnp.array([train_loss, validation_loss])

    _, early_stop = early_stop.update(validation_loss)
    if early_stop.should_stop:
      logger.info("early stopping criterion found")
      break
    if validation_loss < best_loss:
      best_loss = validation_loss
      best_params = params.copy()

  stacked_losses = jnp.vstack(losses)[: (i + 1), :]
  return best_params, stacked_losses
