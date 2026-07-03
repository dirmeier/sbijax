"""Neural ratio estimation.

Implements the contrastive NRE method of :cite:t:`miller2022contrast` as a
functional estimator. The network is a classifier whose logits define a
likelihood-to-evidence ratio; the posterior is obtained by combining the ratio
with the prior and drawing samples with the injected MCMC sampler.
"""

# ruff: noqa: PLR0913
from functools import partial
from typing import NamedTuple

import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from jax._src.flatten_util import ravel_pytree

from sbijax._src.inference._estimator import Estimator, next_round
from sbijax._src.mcmc.nuts import sample_with_nuts
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.train import train_loop


class NREInfo(NamedTuple):
  """Diagnostics returned by :func:`nre`'s ``fit`` (DR-011).

  Attributes:
      round: the training round; ``fit`` reads this back to advance rounds
      losses: a ``(n_epochs, 2)`` array of train/validation losses
  """

  round: int
  losses: jax.Array


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


def _loss(params, rng_key, model, gamma, num_classes, **batch):
  n, _ = batch["y"].shape
  rng_key1, rng_key2, rng_key = jr.split(rng_key, 3)
  log_marg = _as_logits(params, rng_key1, model, num_classes, **batch)
  log_joint = _as_logits(params, rng_key2, model, num_classes, **batch)
  log_marg = log_marg.reshape(n, num_classes + 1)[:, 1:]
  log_joint = log_joint.reshape(n, num_classes + 1)[:, :-1]
  loss = _marginal_joint_loss(gamma, num_classes, log_marg, log_joint)
  return -jnp.mean(loss)


def nre(prior, network, *, sampler=sample_with_nuts, num_classes=10, gamma=1.0):
  """Construct a neural ratio estimator.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      network: a classifier network mapping ``concat(y, theta)`` to logits
      sampler: an MCMC sampler used to draw from the posterior; defaults to NUTS
      num_classes: number of contrastive classes
      gamma: relative weight of the contrastive classes

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
    n_early_stopping_patience=25,
    n_early_stopping_delta=1e-3,
  ):
    if optimizer is None:
      optimizer = optax.adam(0.003)
    itr_key, rng_key = jr.split(rng_key)
    train_iter, val_iter = as_batch_iterators(
      itr_key, data, batch_size, 1.0 - percentage_data_as_validation_set, True
    )
    init_key, rng_key = jr.split(rng_key)
    init_batch = next(iter(train_iter))
    params = network.init(
      init_key,
      jnp.concatenate([init_batch["y"], init_batch["theta"]], axis=-1),
    )

    def loss_fn(params, rng, **batch):
      return _loss(
        params, rng, network, gamma=gamma, num_classes=num_classes, **batch
      )

    params, losses = train_loop(
      rng_key,
      params=params,
      optimizer=optimizer,
      loss_fn=loss_fn,
      validation_loss_fn=loss_fn,
      train_iter=train_iter,
      val_iter=val_iter,
      n_iter=n_iter,
      n_early_stopping_patience=n_early_stopping_patience,
      n_early_stopping_delta=n_early_stopping_delta,
    )
    return params, NREInfo(round=next_round(info), losses=losses)

  def sample(
    rng_key,
    params,
    observable,
    *,
    n_chains=4,
    n_samples=2_000,
    n_warmup=1_000,
    **kwargs,
  ):
    """Draw posterior samples via MCMC.

    Returns:
        a tuple ``(samples, info)`` of the named posterior pytree and an
        ``MCMCSampleInfo``
    """
    observable = jnp.atleast_2d(observable)
    classifier = partial(network.apply, params, is_training=False)

    def log_density(theta):
      lp_prior = prior.log_prob(theta)
      theta_flat, _ = ravel_pytree(theta)
      theta_flat = theta_flat.reshape(observable.shape[0], -1)
      lp = classifier(jnp.concatenate([observable, theta_flat], axis=-1))
      return jnp.sum(lp_prior) + jnp.sum(lp)

    samples, info = sampler(
      rng_key=rng_key,
      lp=log_density,
      prior=prior,
      n_chains=n_chains,
      n_samples=n_samples,
      n_warmup=n_warmup,
      **kwargs,
    )
    return samples, info

  return Estimator(fit=fit, sample=sample)
