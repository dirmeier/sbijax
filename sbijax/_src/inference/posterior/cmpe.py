"""Consistency model posterior estimation.

Implements the CMPE algorithm of :cite:t:`schmitt2023con` as a functional
estimator. Unlike the other trainable methods, CMPE maintains an EMA target
network that is updated after each optimizer step, so ``fit`` runs its own loop
rather than the shared ``train_loop``. Sampling reuses the flow-rejection
sampler shared with FMPE.
"""

# ruff: noqa: PLR0913
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.inference._estimator import Estimator, next_round
from sbijax._src.inference.posterior._sampling import rejection_sample_flow
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.early_stopping import EarlyStopping


class CMPEInfo(NamedTuple):
  """Diagnostics returned by :func:`cmpe`'s ``fit`` (DR-011).

  Attributes:
      round: the training round; ``fit`` reads this back to advance rounds
      losses: a ``(n_epochs, 2)`` array of train/validation losses
  """

  round: int
  losses: jax.Array


def _alpha_t(time):
  return 1.0 / (_time_schedule(time + 1) - _time_schedule(time))


def _time_schedule(n, rho=7, t_min=0.001, t_max=50, n_inters=1000):
  left = t_min ** (1 / rho)
  right = t_max ** (1 / rho) - t_min ** (1 / rho)
  right = (n - 1) / (n_inters - 1) * right
  return (left + right) ** rho


def _discretization_schedule(n_iter, max_iter=1000):
  s0, s1 = 10, 50
  nk = (
    (n_iter / max_iter) * (jnp.square(s1 + 1) - jnp.square(s0))
    + jnp.square(s0)
    - 1
  )
  nk = jnp.ceil(jnp.sqrt(nk)) + 1
  return nk


def _consistency_loss(
  params,
  ema_params,
  rng_key,
  apply_fn,
  n_iter,
  t_min,
  t_max,
  is_training=False,
  **batch,
):
  theta = batch["theta"]
  nk = _discretization_schedule(n_iter)

  t_key, rng_key = jr.split(rng_key)
  time_idx = jr.randint(t_key, shape=(theta.shape[0],), minval=1, maxval=nk - 1)
  tn = _time_schedule(time_idx, t_min=t_min, t_max=t_max, n_inters=nk).reshape(
    -1, 1
  )
  tnp1 = _time_schedule(
    time_idx + 1, t_min=t_min, t_max=t_max, n_inters=nk
  ).reshape(-1, 1)

  noise_key, rng_key = jr.split(rng_key)
  noise = jr.normal(noise_key, shape=(*theta.shape,))

  train_rng, rng_key = jr.split(rng_key)
  fnp1 = apply_fn(
    params,
    train_rng,
    method="vector_field",
    theta=theta + tnp1 * noise,
    time=tnp1,
    context=batch["y"],
    is_training=is_training,
  )
  fn = apply_fn(
    ema_params,
    train_rng,
    method="vector_field",
    theta=theta + tn * noise,
    time=tn,
    context=batch["y"],
    is_training=is_training,
  )
  mse = jnp.sqrt(jnp.mean(jnp.square(fnp1 - fn), axis=1))
  loss = _alpha_t(time_idx) * mse
  return jnp.mean(loss)


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

    losses = jnp.vstack(losses)[: (i + 1), :]
    return best_params, CMPEInfo(round=next_round(info), losses=losses)

  def sample(rng_key, params, observable, *, n_samples=4_000, **kwargs):
    return rejection_sample_flow(
      rng_key, network, params, prior, observable, n_samples
    )

  return Estimator(fit=fit, sample=sample)
