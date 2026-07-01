"""Neural posterior estimation.

Implements the (amortized, single-round) NPE objective of
:cite:t:`greenberg2019automatic` as a functional estimator. The network models
the posterior directly in an unconstrained space via the prior's event-space
bijector; ``sample`` draws from the flow and rejects draws outside the prior
support. The atomic, multi-round variant is expressed through the sequential
driver rather than baked into the estimator.
"""

# ruff: noqa: PLR0913
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.inference._estimator import Estimator
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.train import train_loop


def npe(prior, network, *, use_event_space_bijections=True):
  """Construct a neural posterior estimator.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      network: a conditional density estimator with ``log_prob`` and ``sample``
          methods modelling the posterior
      use_event_space_bijections: if True, train in the prior's unconstrained
          event space

  Returns:
      an :class:`~sbijax._src.inference._estimator.Estimator`
  """
  _, unravel_fn = ravel_pytree(prior.sample(seed=jr.PRNGKey(1)))
  bijector = None
  if use_event_space_bijections and hasattr(
    prior, "experimental_default_event_space_bijector"
  ):
    bijector = prior.experimental_default_event_space_bijector()

  def fit(
    rng_key,
    data,
    *,
    optimizer=None,
    n_iter=1000,
    batch_size=100,
    percentage_data_as_validation_set=0.1,
    n_early_stopping_patience=10,
    n_early_stopping_delta=1e-3,
  ):
    if optimizer is None:
      optimizer = optax.adam(0.0003)
    itr_key, rng_key = jr.split(rng_key)
    train_iter, val_iter = as_batch_iterators(
      itr_key, data, batch_size, 1.0 - percentage_data_as_validation_set, True
    )
    init_key, rng_key = jr.split(rng_key)
    init_batch = next(iter(train_iter))
    params = network.init(
      init_key,
      method="log_prob",
      y=init_batch["theta"],
      x=init_batch["y"],
    )

    def loss_fn(params, rng, **batch):  # noqa: ARG001
      theta, y = batch["theta"], batch["y"]
      log_det = 0.0
      if bijector is not None:
        theta_map = jax.vmap(unravel_fn)(theta)
        theta = bijector.inverse(theta_map)
        log_det = bijector.inverse_log_det_jacobian(theta_map)
        theta = jax.vmap(lambda x: ravel_pytree(x)[0])(theta)
      lp = network.apply(params, None, method="log_prob", y=theta, x=y)
      return -jnp.mean(lp + log_det)

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
    n_samples=4_000,
    check_proposal_probs=True,
    **kwargs,
  ):
    observable = jnp.atleast_2d(observable)
    thetas = None
    n_curr = n_samples
    while n_curr > 0:
      n_sim = 200
      sample_key, rng_key = jr.split(rng_key)
      proposal = network.apply(
        params,
        sample_key,
        method="sample",
        sample_shape=(n_sim,),
        x=jnp.tile(observable, [n_sim, 1]),
      )
      if bijector is not None:
        proposal = bijector.forward(jax.vmap(unravel_fn)(proposal))
        proposal_probs = prior.log_prob(proposal)
        proposal = jax.vmap(lambda x: ravel_pytree(x)[0])(proposal)
      else:
        proposal_probs = prior.log_prob(jax.vmap(unravel_fn)(proposal))
      if check_proposal_probs:
        proposal = proposal[jnp.isfinite(proposal_probs)]
      thetas = proposal if thetas is None else jnp.vstack([thetas, proposal])
      n_curr -= proposal.shape[0]

    def reshape(p):
      if p.ndim == 1:
        p = p.reshape(p.shape[0], 1)
      return p.reshape(1, *p.shape)

    thetas = jax.tree_util.tree_map(
      reshape, jax.vmap(unravel_fn)(thetas[:n_samples])
    )
    return as_inference_data(thetas, jnp.squeeze(observable))

  return Estimator(fit=fit, sample=sample)
