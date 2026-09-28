"""Consistency model posterior estimation (functional objective, design B).

Implements the CMPE algorithm of :cite:t:`schmitt2023con`: consistency training
with a stop-gradient teacher, a pseudo-Huber distance and a time grid that is
refined from ``s0`` to ``s1`` intervals over training. The loss itself is the
network's ``loss`` method (see :func:`~sbijax.nn.make_cm`). The refinement
needs the step count, which the shared ``train`` driver does not pass, so it
is carried next to the optimizer state:

    state.opt_state == CMPEOptState(opt_state=..., step=...)
"""

import math
from typing import Any, NamedTuple

import jax
import optax
from jax import numpy as jnp

from sbijax._src.inference.posterior._sampling import rejection_sample_flow
from sbijax._src.train._types import ObjectiveFns, TrainFns, TrainingState


class CMPEOptState(NamedTuple):
  """Optimizer state of a CMPE objective.

  Attributes:
      opt_state: the optax optimizer state
      step: the number of gradient steps taken so far
  """

  opt_state: Any
  step: jax.Array


def discretization_schedule(
  step: jax.Array, n_train_steps: int, s0: int, s1: int
) -> jax.Array:
  """Return the number of time grid intervals at a training step.

  Implements ``N(k) - 1 = min(s0 * 2^floor(k / K'), s1)`` with
  ``K' = floor(K / (log2(floor(s1 / s0)) + 1))`` from :cite:t:`schmitt2023con`.

  Args:
      step: the current gradient step ``k``
      n_train_steps: the total number of gradient steps ``K``
      s0: initial number of intervals
      s1: final number of intervals

  Returns:
      the number of intervals as an integer array
  """
  steps_per_stage = max(
    math.floor(n_train_steps / (math.log2(s1 // s0) + 1)), 1
  )
  n_doublings = jnp.floor(step / steps_per_stage)
  n_intervals = jnp.minimum(s0 * 2.0**n_doublings, s1)
  return n_intervals.astype(jnp.int32)


def cmpe(
  network, *, n_train_steps: int = 10_000, s0: int = 10, s1: int = 50
) -> ObjectiveFns:
  """Construct a consistency model posterior objective.

  Args:
      network: a consistency model with ``loss`` and ``sample`` methods
          (e.g. from :func:`~sbijax.nn.make_cm`)
      n_train_steps: expected number of gradient steps, i.e., epochs times
          batches per epoch. The time grid reaches ``s1`` intervals after
          this many steps and stays there
      s0: initial number of time grid intervals
      s1: final number of time grid intervals

  Returns:
      an ``ObjectiveFns``

  Raises:
      ValueError: if ``s0 < 1`` or ``s1 < s0``
  """
  if s0 < 1 or s1 < s0:
    raise ValueError(f"need 1 <= s0 <= s1, got s0={s0} and s1={s1}")

  def _loss(params, rng_key, batch, n_intervals, is_training):
    loss = network.apply(
      params,
      rng_key,
      method="loss",
      inputs=batch["theta"],
      context=batch["y"],
      n_intervals=n_intervals,
      is_training=is_training,
    )
    return jnp.mean(loss)

  def _n_intervals(step):
    return discretization_schedule(step, n_train_steps, s0, s1)

  def init_fn(optimizer, rng_key, batch):
    params = network.init(
      rng_key,
      method="loss",
      inputs=batch["theta"],
      context=batch["y"],
      n_intervals=s0,
      is_training=False,
    )
    opt_state = CMPEOptState(optimizer.init(params), jnp.zeros((), jnp.int32))
    return TrainingState(params=params, opt_state=opt_state)

  def step_fn(optimizer, rng_key, state, batch):
    step = state.opt_state.step
    loss, grads = jax.value_and_grad(_loss)(
      state.params, rng_key, batch, _n_intervals(step), True
    )
    updates, opt_state = optimizer.update(
      grads, state.opt_state.opt_state, state.params
    )
    params = optax.apply_updates(state.params, updates)
    return {"loss": loss}, TrainingState(
      params, CMPEOptState(opt_state, step + 1)
    )

  def eval_fn(rng_key, state, batch):
    # a fixed grid keeps validation losses comparable across stages
    return {"loss": _loss(state.params, rng_key, batch, s1, False)}

  def sample_fn(
    rng_key, params, observable, *, n_samples=4_000, prior=None, **kwargs
  ):
    return rejection_sample_flow(
      rng_key, network, params, observable, n_samples, prior
    )

  return ObjectiveFns(TrainFns(init_fn, step_fn, eval_fn), sample_fn)
