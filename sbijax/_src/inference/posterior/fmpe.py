"""Flow matching posterior estimation (functional objective, design B)."""

import jax
import optax
from jax import numpy as jnp

from sbijax._src.inference.posterior._sampling import rejection_sample_flow
from sbijax._src.train._types import ObjectiveFns, TrainFns, TrainingState


def fmpe(network):
  """Construct a flow-matching posterior objective.

  Args:
      network: a continuous normalizing flow with ``loss`` and ``sample``
          methods

  Returns:
      an :class:`~sbijax._src.train._types.ObjectiveFns`
  """

  def _loss(params, rng, batch, is_training):
    lp = network.apply(
      params,
      rng=rng,
      method="loss",
      inputs=batch["theta"],
      context=batch["y"],
      is_training=is_training,
    )
    return jnp.mean(lp)

  def init_fn(optimizer, rng_key, batch):
    params = network.init(
      rng_key,
      method="loss",
      inputs=batch["theta"],
      context=batch["y"],
      is_training=False,
    )
    return TrainingState(params=params, opt_state=optimizer.init(params))

  def step_fn(optimizer, rng_key, state, batch):
    loss, grads = jax.value_and_grad(_loss)(state.params, rng_key, batch, True)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    return {"loss": loss}, TrainingState(
      optax.apply_updates(state.params, updates), opt_state
    )

  def eval_fn(rng_key, state, batch):
    return {"loss": _loss(state.params, rng_key, batch, False)}

  def sample_fn(rng_key, params, observable, *, n_samples=4_000, **kwargs):
    return rejection_sample_flow(
      rng_key, network, params, observable, n_samples
    )

  return ObjectiveFns(TrainFns(init_fn, step_fn, eval_fn), sample_fn)
