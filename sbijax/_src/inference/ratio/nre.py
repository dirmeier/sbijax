"""Neural ratio estimation.

Implements the contrastive NRE method of :cite:t:`miller2022contrast` as a
functional estimator. The network is a classifier whose logits define a
likelihood-to-evidence ratio; the posterior is obtained by combining the ratio
with the prior and drawing samples with the injected MCMC sampler.
"""

# ruff: noqa: PLR0913
from functools import partial

import optax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.inference._estimator import Estimator
from sbijax._src.mcmc import sample_with_nuts
from sbijax._src.nre import _loss
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.train import train_loop


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

    return train_loop(
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
    observable = jnp.atleast_2d(observable)
    classifier = partial(network.apply, params, is_training=False)

    def log_density(theta):
      lp_prior = prior.log_prob(theta)
      theta_flat, _ = ravel_pytree(theta)
      theta_flat = theta_flat.reshape(observable.shape[0], -1)
      lp = classifier(jnp.concatenate([observable, theta_flat], axis=-1))
      return jnp.sum(lp_prior) + jnp.sum(lp)

    samples = sampler(
      rng_key=rng_key,
      lp=log_density,
      prior=prior,
      n_chains=n_chains,
      n_samples=n_samples,
      n_warmup=n_warmup,
      **kwargs,
    )
    return as_inference_data(samples, jnp.squeeze(observable))

  return Estimator(fit=fit, sample=sample)
