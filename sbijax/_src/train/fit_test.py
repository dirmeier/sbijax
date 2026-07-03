import jax
import jax.numpy as jnp
import optax
from jax import random as jr

from sbijax._src.train._types import Info, ObjectiveFns, TrainFns, TrainingState
from sbijax._src.train.fit import fit


def _toy_objective():
  def _loss(params, batch):
    return jnp.mean((batch["y"] @ params["w"] - batch["theta"][:, 0]) ** 2)

  def init_fn(optimizer, rng, batch):
    params = {"w": jnp.zeros((batch["y"].shape[-1],))}
    return TrainingState(params=params, opt_state=optimizer.init(params))

  def step_fn(optimizer, rng, state, batch):
    loss, grads = jax.value_and_grad(_loss)(state.params, batch)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    return {"loss": loss}, TrainingState(optax.apply_updates(state.params, updates), opt_state)

  def eval_fn(rng, state, batch):
    return {"loss": _loss(state.params, batch)}

  return ObjectiveFns(TrainFns(init_fn, step_fn, eval_fn), sample_fn=None)


def test_fit_trains_and_advances_round():
  y = jr.normal(jr.key(0), (512, 3))
  data = {"y": y, "theta": (y @ jnp.array([1.0, -2.0, 0.5]))[:, None]}
  opt = optax.adam(1e-2)
  params, info = fit(
    jr.key(1), _toy_objective(), data,
    optimizer=opt, n_iter=60, batch_size=64,
  )
  assert isinstance(info, Info) and info.round == 0
  assert info.losses.ndim == 2 and info.losses.shape[1] == 2
  assert jnp.allclose(params["w"], jnp.array([1.0, -2.0, 0.5]), atol=0.2)
  _, info1 = fit(
    jr.key(1), _toy_objective(), data,
    optimizer=opt, n_iter=2, batch_size=64, info=info,
  )
  assert info1.round == 1
