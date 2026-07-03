"""Neural posterior estimation.

Implements the NPE objective of :cite:t:`greenberg2019automatic` as a functional
estimator. The network models the posterior directly in an unconstrained space
via the prior's event-space bijector; ``sample`` draws from the flow and rejects
draws outside the prior support. In round 0 the network is trained by maximum
likelihood; in later rounds (driven by :func:`sbijax.run_sequential`) ``fit``
switches to the atomic proposal-posterior loss, selected from the round carried
in the ``Info`` threaded back into ``fit``.
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
from sbijax._src.inference._sample_info import DirectSampleInfo
from sbijax._src.util.dataloader import as_batch_iterators
from sbijax._src.util.train import train_loop


def _to_unconstrained(theta, bijector, unravel_fn):
  """Map raw (constrained) draws to the space the network models.

  Returns the unconstrained parameters passed to the network together with the
  change-of-variables log-determinant, so ``network.log_prob + log_det`` is the
  log posterior density in the constrained space.
  """
  if bijector is None:
    return theta, jnp.zeros(theta.shape[0])
  theta_map = jax.vmap(unravel_fn)(theta)
  theta_u = jax.vmap(lambda x: ravel_pytree(x)[0])(bijector.inverse(theta_map))
  # broadcast to per-sample: an identity bijector returns a scalar ldj
  log_det = jnp.broadcast_to(
    bijector.inverse_log_det_jacobian(theta_map), (theta.shape[0],)
  )
  return theta_u, log_det


def _maximum_likelihood_loss(
  params, _rng, network, bijector, unravel_fn, **batch
):
  """Round-0 loss: maximum likelihood against draws from the prior."""
  theta_u, log_det = _to_unconstrained(batch["theta"], bijector, unravel_fn)
  lp = network.apply(params, None, method="log_prob", y=theta_u, x=batch["y"])
  return -jnp.mean(lp + log_det)


def _atomic_loss(
  params, rng, network, prior, bijector, unravel_fn, num_atoms, **batch
):
  """Round-``> 0`` atomic proposal-posterior loss of NPE-C / APT.

  Corrects for the proposal no longer being the prior by contrasting the true
  parameter against ``num_atoms - 1`` others drawn from the batch, reweighted
  by the prior. Needs only the network, prior and ``num_atoms`` -- no proposal
  density (:cite:t:`greenberg2019automatic`).
  """
  theta, y = batch["theta"], batch["y"]
  n = theta.shape[0]
  m = min(num_atoms, n)
  theta_u, log_det = _to_unconstrained(theta, bijector, unravel_fn)
  # for each row, draw m-1 contrasting rows without replacement (exclude self)
  probs = jnp.ones((n, n)) * (1.0 - jnp.eye(n)) / (n - 1.0)
  choices = jax.vmap(
    lambda key, p: jr.choice(key, n, (m - 1,), replace=False, p=p)
  )(jr.split(rng, n), probs)
  idx = jnp.concatenate([jnp.arange(n)[:, None], choices], axis=1)
  lp_net = network.apply(
    params,
    None,
    method="log_prob",
    y=theta_u[idx].reshape(n * m, -1),
    x=jnp.repeat(y, m, axis=0),
  ).reshape(n, m)
  lp_post = lp_net + log_det[idx]
  lp_prior = prior.log_prob(
    jax.vmap(unravel_fn)(theta[idx].reshape(n * m, -1))
  ).reshape(n, m)
  # importance-reweighted contrast; the true theta sits at atom index 0
  unnormalized = lp_post - lp_prior
  log_prob = unnormalized[:, 0] - jsp.special.logsumexp(unnormalized, axis=-1)
  return -jnp.mean(log_prob)


class NPEInfo(NamedTuple):
  """Diagnostics returned by :func:`npe`'s ``fit`` (DR-011).

  Attributes:
      round: the training round; ``fit`` reads this back to advance rounds and
          to select the loss (round 0 is maximum-likelihood, round > 0 is the
          atomic proposal-posterior loss)
      losses: a ``(n_epochs, 2)`` array of train/validation losses
      num_atoms: the number of atoms used in the contrastive loss this round;
          ``0`` in round 0, where the atomic loss is not used
  """

  round: int
  losses: jax.Array
  num_atoms: int


def npe(prior, network, *, num_atoms=10, use_event_space_bijections=True):
  """Construct a neural posterior estimator.

  In round 0 the network is trained by maximum likelihood against draws from
  the prior. In later rounds (driven by :func:`sbijax.run_sequential`, which
  simulates from the current posterior) ``fit`` switches to the atomic
  proposal-posterior loss of :cite:t:`greenberg2019automatic`, correcting for
  the proposal no longer being the prior. The atomic loss needs only the
  network, the prior and ``num_atoms`` -- no proposal density is threaded in.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      network: a conditional density estimator with ``log_prob`` and ``sample``
          methods modelling the posterior
      num_atoms: the number of atoms in the contrastive proposal-posterior loss
          used in rounds > 0
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
    info=None,
    optimizer=None,
    n_iter=1000,
    batch_size=100,
    percentage_data_as_validation_set=0.1,
    n_early_stopping_patience=10,
    n_early_stopping_delta=1e-3,
  ):
    if optimizer is None:
      optimizer = optax.adam(0.0003)
    rnd = next_round(info)
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

    if rnd > 0:
      loss_fn = partial(
        _atomic_loss,
        network=network,
        prior=prior,
        bijector=bijector,
        unravel_fn=unravel_fn,
        num_atoms=num_atoms,
      )
    else:
      loss_fn = partial(
        _maximum_likelihood_loss,
        network=network,
        bijector=bijector,
        unravel_fn=unravel_fn,
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
    return params, NPEInfo(
      round=rnd, losses=losses, num_atoms=0 if rnd == 0 else num_atoms
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
    return thetas, DirectSampleInfo(n_samples=n_samples)

  return Estimator(fit=fit, sample=sample)
