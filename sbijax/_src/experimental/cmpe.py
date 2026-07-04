"""Consistency model posterior estimation (functional objective, design B).

Implements the CMPE algorithm of :cite:t:`schmitt2023con`.  The consistency
loss requires an EMA copy of the network weights; both the live params and the
EMA params are stored together as a dict under ``TrainingState.params`` so the
shared ``fit`` driver needs no changes:

    state.params == {"params": live_params, "ema_params": ema_params}

``sample_fn`` receives this dict and extracts ``params["params"]`` before
calling the flow-rejection sampler.
"""

# ruff: noqa: PLR0913
import jax
import optax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.inference.posterior._sampling import rejection_sample_flow
from sbijax._src.train._types import ObjectiveFns, TrainFns, TrainingState


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


def cmpe(network, *, t_min=0.001, t_max=50.0):
  """Construct a consistency model posterior objective.

  The returned ``ObjectiveFns`` is trained via
  the shared :func:`~sbijax.train` driver.  EMA params are
  threaded through ``TrainingState.params`` as a dict:
  ``{"params": live_params, "ema_params": ema_params}``.

  Args:
      network: a consistency model with ``vector_field`` and ``sample``
          methods
      t_min: minimal time point for ODE integration
      t_max: maximal time point for ODE integration

  Returns:
      an ``ObjectiveFns``
  """

  def _loss(params_dict, rng_key, batch, is_training):
    params = params_dict["params"]
    ema_params = params_dict["ema_params"]
    theta = batch["theta"]
    # n_iter fixed at reference value so the schedule is consistent across fit.
    n_iter = 1001
    nk = _discretization_schedule(n_iter)

    t_key, rng_key = jr.split(rng_key)
    time_idx = jr.randint(
      t_key, shape=(theta.shape[0],), minval=1, maxval=nk - 1
    )
    tn = _time_schedule(
      time_idx, t_min=t_min, t_max=t_max, n_inters=nk
    ).reshape(-1, 1)
    tnp1 = _time_schedule(
      time_idx + 1, t_min=t_min, t_max=t_max, n_inters=nk
    ).reshape(-1, 1)

    noise_key, rng_key = jr.split(rng_key)
    noise = jr.normal(noise_key, shape=(*theta.shape,))

    train_rng, rng_key = jr.split(rng_key)
    fnp1 = network.apply(
      params,
      train_rng,
      method="vector_field",
      theta=theta + tnp1 * noise,
      time=tnp1,
      context=batch["y"],
      is_training=is_training,
    )
    fn = network.apply(
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

  def init_fn(optimizer, rng_key, batch):
    times = jr.uniform(rng_key, shape=(batch["y"].shape[0], 1))
    params = network.init(
      rng_key,
      method="vector_field",
      theta=batch["theta"],
      time=times,
      context=batch["y"],
      is_training=True,
    )
    params_dict = {"params": params, "ema_params": params}
    return TrainingState(params=params_dict, opt_state=optimizer.init(params))

  def step_fn(optimizer, rng_key, state, batch):
    loss, grads = jax.value_and_grad(_loss)(state.params, rng_key, batch, True)
    # grad is only valid for live params; zero out ema_params gradient
    live_grads = grads["params"]
    updates, opt_state = optimizer.update(
      live_grads, state.opt_state, state.params["params"]
    )
    new_params = optax.apply_updates(state.params["params"], updates)
    new_ema = optax.incremental_update(
      new_params, state.params["ema_params"], step_size=0.01
    )
    new_params_dict = {"params": new_params, "ema_params": new_ema}
    return {"loss": loss}, TrainingState(new_params_dict, opt_state)

  def eval_fn(rng_key, state, batch):
    return {"loss": _loss(state.params, rng_key, batch, False)}

  def sample_fn(rng_key, params, observable, *, n_samples=4_000, **kwargs):
    # params is the params_dict; sampling uses only the live params.
    live_params = params["params"]
    return rejection_sample_flow(
      rng_key, network, live_params, observable, n_samples
    )

  return ObjectiveFns(TrainFns(init_fn, step_fn, eval_fn), sample_fn)
