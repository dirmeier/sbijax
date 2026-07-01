"""Consistency model posterior estimation.

Implements the CMPE algorithm of :cite:t:`schmitt2023con` as a functional
estimator. Unlike the other trainable methods, CMPE maintains an EMA target
network that is updated after each optimizer step, so ``fit`` runs its own loop
rather than the shared ``train_loop``. Sampling reuses the flow-rejection
sampler shared with FMPE.
"""

# ruff: noqa: PLR0913
from functools import partial

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.cmpe import _consistency_loss
from sbijax._src.inference._estimator import Estimator
from sbijax._src.inference.posterior._sampling import rejection_sample_flow
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.early_stopping import EarlyStopping


def cmpe(prior, network, *, t_min=0.001, t_max=50.0):
  """Construct a consistency model posterior estimator.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      network: a consistency model with ``vector_field`` and ``sample`` methods
      t_min: minimal time point for ODE integration
      t_max: maximal time point for ODE integration

  Returns:
      an :class:`~sbijax._src.inference._estimator.Estimator`
  """

  def fit(
    rng_key,
    data,
    *,
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
    times = jr.uniform(jr.PRNGKey(0), shape=(init_batch["y"].shape[0], 1))
    params = network.init(
      init_key,
      method="vector_field",
      theta=init_batch["theta"],
      time=times,
      context=init_batch["y"],
      is_training=True,
    )
    ema_params = params.copy()
    state = optimizer.init(params)

    train_loss_fn = jax.jit(
      partial(
        _consistency_loss,
        apply_fn=network.apply,
        is_training=True,
        t_max=t_max,
        t_min=t_min,
      )
    )
    val_loss_fn = jax.jit(
      partial(
        _consistency_loss,
        apply_fn=network.apply,
        is_training=False,
        t_max=t_max,
        t_min=t_min,
        n_iter=n_iter,
      )
    )

    @jax.jit
    def step(params, ema_params, rng, state, **batch):
      loss, grads = jax.value_and_grad(train_loss_fn)(
        params, ema_params, rng, n_iter=n_iter + 1, **batch
      )
      updates, new_state = optimizer.update(grads, state, params)
      new_params = optax.apply_updates(params, updates)
      new_ema = optax.incremental_update(new_params, ema_params, step_size=0.01)
      return loss, new_params, new_ema, new_state

    def validation_loss(rng_key, params, ema_params):
      total = 0.0
      for batch in val_iter:
        val_key, rng_key = jr.split(rng_key)
        total += val_loss_fn(params, ema_params, val_key, **batch) * (
          batch["y"].shape[0] / val_iter.num_samples
        )
      return total

    losses = np.zeros([n_iter, 2])
    early_stop = EarlyStopping(
      n_early_stopping_delta, n_early_stopping_patience
    )
    best_params, best_loss = None, np.inf
    i = 0
    for i in range(n_iter):
      train_loss = 0.0
      epoch_key = jr.fold_in(rng_key, i)
      for batch in train_iter:
        train_key, epoch_key = jr.split(epoch_key)
        batch_loss, params, ema_params, state = step(
          params, ema_params, train_key, state, **batch
        )
        train_loss += batch_loss * (
          batch["y"].shape[0] / train_iter.num_samples
        )
      val_key, epoch_key = jr.split(epoch_key)
      val_loss = validation_loss(val_key, params, ema_params)
      losses[i] = jnp.array([train_loss, val_loss])
      _, early_stop = early_stop.update(val_loss)
      if early_stop.should_stop:
        break
      if val_loss < best_loss:
        best_loss = val_loss
        best_params = params.copy()

    return best_params, jnp.vstack(losses)[: (i + 1), :]

  def sample(rng_key, params, observable, *, n_samples=4_000, **kwargs):
    return rejection_sample_flow(
      rng_key, network, params, prior, observable, n_samples
    )

  return Estimator(fit=fit, sample=sample)
