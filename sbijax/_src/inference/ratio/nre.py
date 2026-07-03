"""Neural ratio estimation.

Implements the contrastive NRE method of :cite:t:`miller2022contrast` as a
functional objective. The network is a classifier whose logits define a
likelihood-to-evidence ratio; the posterior is obtained at sample time by
handing the ratio log-density to an injected MCMC sampler that adds the prior.
"""

# ruff: noqa: PLR0913

from functools import partial

import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from jax._src.flatten_util import ravel_pytree

from sbijax._src.train._types import ObjectiveFns, TrainFns, TrainingState


def _get_prior_probs_marginal_and_joint(k, gamma):
  p_marginal = 1 / (1 + gamma * k)
  p_joint = gamma / (1 + gamma * k)
  return p_marginal, p_joint


def _as_logits(params, rng_key, model, k, theta, y):
  n = theta.shape[0]
  y = jnp.repeat(y, k + 1, axis=0)
  ps = jnp.ones((n, n)) * (1.0 - jnp.eye(n)) / (n - 1.0)
  choices = jax.vmap(
    lambda key, p: jr.choice(key, n, (k,), replace=False, p=p)
  )(jr.split(rng_key, n), ps)
  contrasting_theta = theta[choices]
  atomic_theta = jnp.concatenate(
    [theta[:, None, :], contrasting_theta], axis=1
  ).reshape(n * (k + 1), -1)
  inputs = jnp.concatenate([y, atomic_theta], axis=-1)
  return model.apply(params, inputs, is_training=False)


def _marginal_joint_loss(gamma, num_classes, log_marg, log_joint):
  loggamma = jnp.log(gamma)
  log_k = jnp.full((log_marg.shape[0], 1), jnp.log(num_classes))
  denominator_marginal = jnp.concatenate([loggamma + log_marg, log_k], axis=-1)
  denominator_joint = jnp.concatenate([loggamma + log_joint, log_k], axis=-1)
  log_prob_marginal = log_k - jsp.special.logsumexp(
    denominator_marginal, axis=-1
  )
  log_prob_joint = (
    loggamma
    + log_joint[:, 0]
    - jsp.special.logsumexp(denominator_joint, axis=-1)
  )
  p_marg, p_joint = _get_prior_probs_marginal_and_joint(num_classes, gamma)
  return p_marg * log_prob_marginal + p_joint * num_classes * log_prob_joint


def _classifier_loss(params, rng_key, model, gamma, num_classes, **batch):
  n, _ = batch["y"].shape
  rng_key1, rng_key2, _ = jr.split(rng_key, 3)
  log_marg = _as_logits(params, rng_key1, model, num_classes, **batch)
  log_joint = _as_logits(params, rng_key2, model, num_classes, **batch)
  log_marg = log_marg.reshape(n, num_classes + 1)[:, 1:]
  log_joint = log_joint.reshape(n, num_classes + 1)[:, :-1]
  loss = _marginal_joint_loss(gamma, num_classes, log_marg, log_joint)
  return -jnp.mean(loss)


def nre(network, *, num_classes=10, gamma=1.0):
  """Construct a neural ratio objective.

  Args:
      network: a classifier network mapping ``concat(y, theta)`` to logits
      num_classes: number of contrastive classes
      gamma: relative weight of the contrastive classes

  Returns:
      an :class:`~sbijax._src.train._types.ObjectiveFns`
  """

  def _loss(params, rng, batch):
    return _classifier_loss(
      params, rng, network, gamma=gamma, num_classes=num_classes, **batch
    )

  def init_fn(optimizer, rng_key, batch):
    params = network.init(
      rng_key,
      jnp.concatenate([batch["y"], batch["theta"]], axis=-1),
    )
    return TrainingState(params=params, opt_state=optimizer.init(params))

  def step_fn(optimizer, rng_key, state, batch):
    loss, grads = jax.value_and_grad(_loss)(state.params, rng_key, batch)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    return {"loss": loss}, TrainingState(
      optax.apply_updates(state.params, updates), opt_state
    )

  def eval_fn(rng_key, state, batch):
    return {"loss": _loss(state.params, rng_key, batch)}

  def sample_fn(
    rng_key,
    params,
    observable,
    *,
    sampler=None,
    n_chains=4,
    n_samples=2_000,
    n_warmup=1_000,
    **kwargs,
  ):
    if sampler is None:
      raise ValueError(
        "nre sampling requires a sampler, e.g. make_sampler(nuts, prior=prior)"
      )
    observable = jnp.atleast_2d(observable)
    classifier = partial(network.apply, params, is_training=False)

    def loglik_fn(theta):
      theta_flat, _ = ravel_pytree(theta)
      theta_flat = theta_flat.reshape(observable.shape[0], -1)
      return classifier(jnp.concatenate([observable, theta_flat], axis=-1))

    return sampler(
      rng_key,
      loglik_fn,
      n_chains=n_chains,
      n_samples=n_samples,
      n_warmup=n_warmup,
    )

  return ObjectiveFns(TrainFns(init_fn, step_fn, eval_fn), sample_fn)
