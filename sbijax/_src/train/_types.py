"""Records shared by the low-level objective API (design B)."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax


class TrainingState(NamedTuple):
  """Carry threaded through ``step_fn`` (cf. a blackjax state).

  Attributes:
      params: the network parameter pytree
      opt_state: the optax optimizer state
  """

  params: Any
  opt_state: Any


class TrainFns(NamedTuple):
  """The low-level training seam. ``fit`` binds the optimizer.

  Attributes:
      init_fn: ``(optimizer, rng_key, batch) -> TrainingState``
      step_fn: ``(optimizer, rng_key, state, batch) -> (metrics, state)``
      eval_fn: ``(rng_key, state, batch) -> metrics``
  """

  init_fn: Callable
  step_fn: Callable
  eval_fn: Callable


class ObjectiveFns(NamedTuple):
  """A trainable posterior/likelihood/ratio estimator.

  Attributes:
      train: the :class:`TrainFns` primitives
      sample_fn: ``(rng_key, params, observable, *, sampler=None, **kwargs) ->
          (samples, info)``
      extra: optional ``(prior) -> ObjectiveFns`` builder for a round > 0
          objective (NPE atomic loss); ``None`` otherwise
  """

  train: TrainFns
  sample_fn: Callable
  extra: Any = None


class SummaryFns(NamedTuple):
  """A trainable summary network, trained by the same generic ``fit``.

  Attributes:
      train: the :class:`TrainFns` primitives
      summarize_fn: ``(params, data) -> summaries``
  """

  train: TrainFns
  summarize_fn: Callable


class Info(NamedTuple):
  """Generic diagnostics from ``fit`` (replaces per-method records).

  Attributes:
      round: the training round; ``fit`` reads this back to advance rounds
      losses: a ``(n_epochs, 2)`` array of train/validation losses
  """

  round: int
  losses: jax.Array


def next_round(info):
  """Return the round a ``fit`` call trains under.

  Args:
      info: the previous round's ``Info``, or ``None`` for round 0

  Returns:
      ``0`` if ``info`` is ``None``, otherwise ``info.round + 1``
  """
  return 0 if info is None else info.round + 1
