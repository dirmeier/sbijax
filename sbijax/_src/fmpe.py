import jax
import optax
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src._ne_base import NE
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.train import train_loop
from sbijax._src.util.types import PyTree


# ruff: noqa: PLR0913
class FMPE(NE):
  r"""Flow matching posterior estimation.

  Implements the FMPE algorithm introduced in :cite:t:`wilderberger2023flow`.

  Args:
      model_fns: a tuple of callables. The first element needs to be a
          function that constructs a tfd.JointDistributionNamed, the second
          element is a simulator function.
      density_estimator: a continuous normalizing flow model

  Examples:
      >>> from sbijax import FMPE
      >>> from sbijax.nn import make_cnf
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      ...
      >>> prior = lambda: tfd.JointDistributionNamed(
      ...     dict(theta=tfd.Normal(0.0, 1.0))
      ... )
      >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
      >>> fns = prior, s
      >>> neural_network = make_cnf(1)
      >>> model = FMPE(fns, neural_network)

  References:
      Wildberger, Jonas, et al. "Flow Matching for Scalable Simulation-Based Inference." Advances in Neural Information Processing Systems, 2024.
  """

  def __init__(self, model_fns, density_estimator):
    super().__init__(model_fns, density_estimator)

  def fit(
    self,
    rng_key: Array,
    data: PyTree,
    *,
    optimizer: optax.GradientTransformation | None = None,
    n_iter: int = 1000,
    batch_size: int = 100,
    percentage_data_as_validation_set: float = 0.1,
    n_early_stopping_patience: int = 10,
    n_early_stopping_delta: float = 0.001,
    **kwargs,
  ):
    """Fit the model.

    Args:
        rng_key: a jax random key
        data: data set obtained from calling
            `simulate_data_and_possibly_append`
        optimizer: an optax optimizer object
        n_iter: maximal number of training iterations per round
        batch_size:  batch size used for training the model
        percentage_data_as_validation_set: percentage of the simulated
            data that is used for validation and early stopping
        n_early_stopping_patience: number of iterations of no improvement
            of training the flow before stopping optimisation
        **kwargs: optional keyword arguments

    Returns:
        a tuple of parameters and a tuple of the training information
    """
    if optimizer is None:
      optimizer = optax.adam(0.0003)
    itr_key, rng_key = jr.split(rng_key)
    train_iter, val_iter = self.as_iterators(
      itr_key, data, batch_size, percentage_data_as_validation_set
    )
    params, losses = self._fit_model_single_round(
      seed=rng_key,
      train_iter=train_iter,
      val_iter=val_iter,
      optimizer=optimizer,
      n_iter=n_iter,
      n_early_stopping_patience=n_early_stopping_patience,
      n_early_stopping_delta=n_early_stopping_delta,
    )

    return params, losses

  def _fit_model_single_round(
    self,
    seed,
    train_iter,
    val_iter,
    optimizer,
    n_iter,
    n_early_stopping_patience,
    n_early_stopping_delta,
  ):
    init_key, seed = jr.split(seed)
    params = self._init_params(init_key, **next(iter(train_iter)))

    def loss_fn(params, rng, **batch):
      lp = self.model.apply(
        params,
        rng=rng,
        method="loss",
        inputs=batch["theta"],
        context=batch["y"],
        is_training=True,
      )
      return jnp.mean(lp)

    def validation_loss_fn(params, rng, **batch):
      lp = self.model.apply(
        params,
        rng=rng,
        method="loss",
        inputs=batch["theta"],
        context=batch["y"],
        is_training=False,
      )
      return jnp.mean(lp)

    return train_loop(
      seed,
      params=params,
      optimizer=optimizer,
      loss_fn=loss_fn,
      validation_loss_fn=validation_loss_fn,
      train_iter=train_iter,
      val_iter=val_iter,
      n_iter=n_iter,
      n_early_stopping_patience=n_early_stopping_patience,
      n_early_stopping_delta=n_early_stopping_delta,
    )

  def _init_params(self, rng_key, **init_data):
    params = self.model.init(
      rng_key,
      method="loss",
      inputs=init_data["theta"],
      context=init_data["y"],
      is_training=False,
    )
    return params

  # ruff: noqa: D417
  def sample_posterior(
    self, rng_key, params, observable, *, n_samples=4_000, **kwargs
  ):
    r"""Sample from the approximate posterior.

    Args:
        rng_key: a jax random key
        params: a pytree of neural network parameters
        observable: observation to condition on
        n_samples: number of samples to draw

    Returns:
        returns an array of samples from the posterior distribution of
        dimension (n_samples \times p)
    """
    observable = jnp.atleast_2d(observable)

    thetas = None
    n_curr = n_samples
    n_total_simulations_round = jnp.asarray(0)
    _, unravel_fn = ravel_pytree(self.prior.sample(seed=jr.PRNGKey(1)))
    while n_curr > 0:
      n_sim = jnp.minimum(1024, jnp.maximum(1024, n_curr))
      n_total_simulations_round += n_sim
      sample_key, rng_key = jr.split(rng_key)
      proposal = self.model.apply(
        params,
        sample_key,
        method="sample",
        context=jnp.tile(observable, [n_sim, 1]),
        is_training=False,
      )
      proposal_probs = self.prior.log_prob(jax.vmap(unravel_fn)(proposal))
      proposal_accepted = proposal[jnp.isfinite(proposal_probs)]
      if thetas is None:
        thetas = proposal_accepted
      else:
        thetas = jnp.vstack([thetas, proposal_accepted])
      n_curr -= proposal_accepted.shape[0]

    assert thetas is not None
    ess = float(thetas.shape[0] / n_total_simulations_round)

    def reshape(p):
      if p.ndim == 1:
        p = p.reshape(p.shape[0], 1)
      p = p.reshape(1, *p.shape)
      return p

    thetas = jax.tree_util.tree_map(
      reshape, jax.vmap(unravel_fn)(thetas[:n_samples])
    )
    inference_data = as_inference_data(thetas, jnp.squeeze(observable))
    return inference_data, ess

  def _simulate_parameters_with_model(
    self, rng_key, params, observable, *, n_samples=4_000, **kwargs
  ):
    return self.sample_posterior(
      rng_key, params, observable, n_samples=n_samples, **kwargs
    )
