"""Neural likelihood estimation.

Implements the method introduced in :cite:t:`papama2019neural` as a functional
objective: a factory that turns a conditional density network into an
:class:`~sbijax._src.train._types.ObjectiveFns`. The network models the
likelihood ``p(y | theta)``; the posterior is obtained at sample time by
handing the likelihood log-density to an injected MCMC sampler that adds the
prior.
"""

import jax
import optax
from jax import numpy as jnp
from jax._src.flatten_util import ravel_pytree

from sbijax._src.train._types import ObjectiveFns, TrainFns, TrainingState


def nle(network):
  """Construct a neural likelihood objective.

  Args:
      network: a conditional density estimator exposing a ``log_prob`` method

  Returns:
      an :class:`~sbijax._src.train._types.ObjectiveFns`
  """

  def _loss(params, rng, batch):  # noqa: ARG001
    lp = network.apply(
      params,
      rng=None,
      method="log_prob",
      y=batch["y"],
      x=batch["theta"],
    )
    return -jnp.mean(lp)

  def init_fn(optimizer, rng_key, batch):
    params = network.init(
      rng_key,
      method="log_prob",
      y=batch["y"],
      x=batch["theta"],
    )
    return TrainingState(params=params, opt_state=optimizer.init(params))

  def step_fn(optimizer, rng_key, state, batch):
    loss, grads = jax.value_and_grad(_loss)(state.params, rng_key, batch)
    updates, opt_state = optimizer.update(
      grads, state.opt_state, state.params
    )
    return {"loss": loss}, TrainingState(
      optax.apply_updates(state.params, updates), opt_state
    )

  def eval_fn(rng_key, state, batch):
    return {"loss": _loss(state.params, rng_key, batch)}

  def sample_fn(
    rng_key, params, observable, *, sampler=None, n_chains=4,
    n_samples=2_000, n_warmup=1_000, **kwargs
  ):
    if sampler is None:
      raise ValueError(
        "nle sampling requires a sampler, e.g. make_sampler(nuts, prior=prior)"
      )
    observable = jnp.atleast_2d(observable)

    def loglik_fn(theta):
      theta_flat, _ = ravel_pytree(theta)
      theta_tiled = jnp.tile(theta_flat, [observable.shape[0], 1])
      return network.apply(params, rng=None, method="log_prob",
                           y=observable, x=theta_tiled)

    return sampler(rng_key, loglik_fn, n_chains=n_chains,
                   n_samples=n_samples, n_warmup=n_warmup)

  return ObjectiveFns(TrainFns(init_fn, step_fn, eval_fn), sample_fn)
