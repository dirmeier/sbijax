# pylint: skip-file

import chex
import optax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.util.dataloader import DataLoader
from sbijax._src.util.train import train_loop


def _quadratic_iters():
  batches = [
    {"y": jnp.ones((4, 1)), "theta": jnp.zeros((4, 2))},
    {"y": jnp.ones((4, 1)), "theta": jnp.zeros((4, 2))},
  ]
  itr = DataLoader(batches, num_samples=8)
  return itr, itr


def test_train_loop_minimises_and_returns_best_params():
  train_iter, val_iter = _quadratic_iters()
  params = {"w": jnp.array([3.0, -2.0])}

  def loss_fn(params, rng, **batch):
    return jnp.sum(params["w"] ** 2)

  best_params, losses = train_loop(
    jr.PRNGKey(0),
    params=params,
    optimizer=optax.sgd(0.1),
    loss_fn=loss_fn,
    validation_loss_fn=loss_fn,
    train_iter=train_iter,
    val_iter=val_iter,
    n_iter=20,
    n_early_stopping_patience=100,
  )

  # convex problem: best params must be closer to the minimum (0) than init
  assert jnp.sum(best_params["w"] ** 2) < jnp.sum(params["w"] ** 2)
  # no early stop -> a row of (train_loss, val_loss) per iteration
  chex.assert_shape(losses, (20, 2))


def test_train_loop_early_stops_on_plateau():
  train_iter, val_iter = _quadratic_iters()
  params = {"w": jnp.array([1.0])}

  def loss_fn(params, rng, **batch):
    return jnp.asarray(5.0)

  _, losses = train_loop(
    jr.PRNGKey(0),
    params=params,
    optimizer=optax.sgd(0.1),
    loss_fn=loss_fn,
    validation_loss_fn=loss_fn,
    train_iter=train_iter,
    val_iter=val_iter,
    n_iter=10,
    n_early_stopping_patience=0,
  )

  # constant loss + zero patience -> stops long before n_iter=10
  assert losses.shape[0] == 2


def test_train_loop_validation_loss_is_sample_weighted():
  # two batches of different sizes: weighted mean must respect batch sizes
  train_batches = [{"y": jnp.ones((6, 1)), "theta": jnp.zeros((6, 1))}]
  val_batches = [
    {"y": jnp.ones((6, 1)), "theta": jnp.zeros((6, 1))},
    {"y": jnp.ones((2, 1)), "theta": jnp.zeros((2, 1))},
  ]
  train_iter = DataLoader(train_batches, num_samples=6)
  val_iter = DataLoader(val_batches, num_samples=8)
  params = {"w": jnp.array([0.0])}

  def loss_fn(params, rng, **batch):
    # per-batch loss = number of rows in the batch
    return jnp.asarray(float(batch["y"].shape[0]))

  _, losses = train_loop(
    jr.PRNGKey(0),
    params=params,
    optimizer=optax.sgd(0.0),
    loss_fn=loss_fn,
    validation_loss_fn=loss_fn,
    train_iter=train_iter,
    val_iter=val_iter,
    n_iter=1,
    n_early_stopping_patience=100,
  )

  # val loss = 6*(6/8) + 2*(2/8) = 4.5 + 0.5 = 5.0
  chex.assert_trees_all_close(losses[0, 1], 5.0, atol=1e-5)
