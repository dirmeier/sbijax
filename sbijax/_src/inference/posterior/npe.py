"""Neural posterior estimation.

Implements the NPE objective of :cite:t:`greenberg2019automatic` as a
functional estimator. The network models the posterior directly in the
parameter space; no event-space bijectors are applied.

In round 0 the network is trained by maximum likelihood (amortized).
In later rounds (driven by :func:`sbijax.run_sequential`), call
``obj.extra(prior)`` to obtain the atomic proposal-posterior objective
(:cite:t:`greenberg2019automatic`) which corrects for the proposal no
longer being the prior.
"""

# ruff: noqa: PLR0913
from functools import partial

import jax
import jax.scipy as jsp
import optax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.inference._sample_info import DirectSampleInfo
from sbijax._src.train._types import ObjectiveFns, TrainFns, TrainingState


def _maximum_likelihood_loss(params, _rng, network, **batch):
  """Round-0 loss: maximum likelihood against draws from the prior.

  Args:
      params: the network parameter pytree
      _rng: unused rng key (kept for a uniform ``(params, rng, **batch)``
          signature)
      network: a conditional density estimator with a ``log_prob`` method
      **batch: must contain ``theta`` (flat ``(n, d)`` array) and ``y``
          (observations ``(n, obs_dim)``)

  Returns:
      a scalar loss
  """
  lp = network.apply(
    params, None, method="log_prob", y=batch["theta"], x=batch["y"]
  )
  return -jnp.mean(lp)


def _atomic_loss(params, rng, network, prior, num_atoms, **batch):
  """Round->0 atomic proposal-posterior loss of NPE-C / APT.

  Corrects for the proposal no longer being the prior by contrasting the
  true parameter against ``num_atoms - 1`` others drawn from the batch,
  reweighted by the prior.  Needs only the network, the prior and
  ``num_atoms`` — no proposal density (:cite:t:`greenberg2019automatic`).

  Args:
      params: the network parameter pytree
      rng: a jax random key
      network: a conditional density estimator with a ``log_prob`` method
      prior: a tfd distribution; only ``log_prob`` is called
      num_atoms: number of atoms in the contrastive loss
      **batch: must contain ``theta`` (flat ``(n, d)`` array) and ``y``
          (observations ``(n, obs_dim)``)

  Returns:
      a scalar loss
  """
  theta, y = batch["theta"], batch["y"]
  n = theta.shape[0]
  m = min(num_atoms, n)

  # Unravel flat theta to the prior's pytree structure for log_prob.
  _, unravel_fn = ravel_pytree(prior.sample(seed=jr.key(0)))

  # For each row draw m-1 contrasting rows without replacement (exclude self).
  probs = jnp.ones((n, n)) * (1.0 - jnp.eye(n)) / (n - 1.0)
  choices = jax.vmap(
    lambda key, p: jr.choice(key, n, (m - 1,), replace=False, p=p)
  )(jr.split(rng, n), probs)
  idx = jnp.concatenate([jnp.arange(n)[:, None], choices], axis=1)

  lp_net = network.apply(
    params,
    None,
    method="log_prob",
    y=theta[idx].reshape(n * m, -1),
    x=jnp.repeat(y, m, axis=0),
  ).reshape(n, m)

  lp_prior = prior.log_prob(
    jax.vmap(unravel_fn)(theta[idx].reshape(n * m, -1))
  ).reshape(n, m)

  # Importance-reweighted contrast; the true theta sits at atom index 0.
  unnormalized = lp_net - lp_prior
  log_prob = unnormalized[:, 0] - jsp.special.logsumexp(unnormalized, axis=-1)
  return -jnp.mean(log_prob)


def npe(network, *, num_atoms=10):
  """Construct a neural posterior estimator.

  In round 0 use the returned :class:`~sbijax._src.train._types.ObjectiveFns`
  directly for amortized maximum-likelihood training. For round > 0 (i.e.
  when training on data simulated from a fitted posterior rather than the
  prior), call ``obj.extra(prior)`` to obtain the atomic proposal-posterior
  objective of :cite:t:`greenberg2019automatic`.

  Args:
      network: a conditional density estimator with ``log_prob`` and
          ``sample`` methods (e.g. from :func:`~sbijax._src.nn.make_flow`)
      num_atoms: the number of atoms in the contrastive proposal-posterior
          loss used by ``extra(prior)``

  Returns:
      an :class:`~sbijax._src.train._types.ObjectiveFns`; its ``extra``
      field is a callable ``(prior) -> ObjectiveFns`` for sequential rounds
  """

  def _objective(loss_fn, extra):
    def init_fn(optimizer, rng_key, batch):
      """Initialise network params and optimizer state from a sample batch.

      Args:
          optimizer: an optax optimizer
          rng_key: a jax random key
          batch: a ``{"theta", "y"}`` batch dict

      Returns:
          a :class:`~sbijax._src.train._types.TrainingState`
      """
      params = network.init(
        rng_key, method="log_prob", y=batch["theta"], x=batch["y"]
      )
      return TrainingState(params=params, opt_state=optimizer.init(params))

    def step_fn(optimizer, rng_key, state, batch):
      """Apply one gradient update.

      Args:
          optimizer: an optax optimizer
          rng_key: a jax random key
          state: the current :class:`~sbijax._src.train._types.TrainingState`
          batch: a ``{"theta", "y"}`` batch dict

      Returns:
          a tuple ``({"loss": scalar}, new_state)``
      """
      loss, grads = jax.value_and_grad(loss_fn)(
        state.params, rng_key, **batch
      )
      updates, opt_state = optimizer.update(
        grads, state.opt_state, state.params
      )
      return {"loss": loss}, TrainingState(
        optax.apply_updates(state.params, updates), opt_state
      )

    def eval_fn(rng_key, state, batch):
      """Evaluate the loss without updating parameters.

      Args:
          rng_key: a jax random key
          state: the current :class:`~sbijax._src.train._types.TrainingState`
          batch: a ``{"theta", "y"}`` batch dict

      Returns:
          ``{"loss": scalar}``
      """
      return {"loss": loss_fn(state.params, rng_key, **batch)}

    def sample_fn(
      rng_key, params, observable, *, sampler=None, n_samples=4_000, **kwargs
    ):
      """Draw posterior samples from the trained flow.

      Args:
          rng_key: a jax random key
          params: the trained network parameters
          observable: a 1-D (or 2-D with one row) observation array
          sampler: unused; present for API symmetry
          n_samples: number of posterior draws to return
          **kwargs: ignored

      Returns:
          a tuple ``(samples, DirectSampleInfo)`` where ``samples`` is a
          named pytree with each leaf shaped ``(1, n_samples, dim)``
      """
      observable = jnp.atleast_2d(observable)
      thetas = network.apply(
        params,
        rng_key,
        method="sample",
        sample_shape=(n_samples,),
        x=jnp.tile(observable, [n_samples, 1]),
      )

      def reshape(p):
        if p.ndim == 1:
          p = p.reshape(p.shape[0], 1)
        return p.reshape(1, *p.shape)

      thetas = jax.tree_util.tree_map(reshape, {"theta": thetas})
      return thetas, DirectSampleInfo(n_samples=n_samples)

    return ObjectiveFns(TrainFns(init_fn, step_fn, eval_fn), sample_fn, extra)

  ml = partial(_maximum_likelihood_loss, network=network)
  atomic = lambda prior: _objective(  # noqa: E731
    partial(_atomic_loss, network=network, prior=prior, num_atoms=num_atoms),
    extra=None,
  )
  return _objective(ml, extra=atomic)
