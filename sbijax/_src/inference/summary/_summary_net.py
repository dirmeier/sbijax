"""The uniform interface for learned summary statistics (design B)."""

import jax
import optax

from sbijax._src.train._types import SummaryFns, TrainFns, TrainingState


def make_summary_net(network, jsd_loss):
  """Build a :class:`SummaryFns` from a network and a JSD summary loss.

  The returned record exposes the same :class:`TrainFns` seam as an
  ``ObjectiveFns`` so the generic ``fit`` trains a summary network unchanged;
  ``summarize_fn`` maps data through the fitted network.

  Args:
      network: a summary network with ``forward``, ``summary`` (and ``critic``)
          methods
      jsd_loss: a callable ``(params, rng, apply_fn, **batch) -> scalar``

  Returns:
      a :class:`~sbijax._src.train._types.SummaryFns`
  """

  def _loss(params, rng, batch):
    return jsd_loss(params, rng, network.apply, **batch)

  def init_fn(optimizer, rng_key, batch):
    params = network.init(rng_key, method="forward", **batch)
    return TrainingState(params=params, opt_state=optimizer.init(params))

  def step_fn(optimizer, rng_key, state, batch):
    loss, grads = jax.value_and_grad(_loss)(state.params, rng_key, batch)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    return {"loss": loss}, TrainingState(
      optax.apply_updates(state.params, updates), opt_state
    )

  def eval_fn(rng_key, state, batch):
    return {"loss": _loss(state.params, rng_key, batch)}

  def summarize_fn(params, data):
    if params is None or len(params) == 0:
      return data
    if isinstance(data, dict):
      ret = data.copy()
      ret["y"] = network.apply(params, method="summary", y=data["y"])
      return ret
    return network.apply(params, method="summary", y=data)

  return SummaryFns(TrainFns(init_fn, step_fn, eval_fn), summarize_fn)
