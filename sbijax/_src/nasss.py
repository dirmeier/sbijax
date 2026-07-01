import jax
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.nass import NASS
from sbijax._src.util.train import train_loop


def _sample_unit_sphere(rng_key, n, dim):
  u = jr.normal(rng_key, (n, dim))
  norm = jnp.linalg.norm(u, ord=2, axis=-1, keepdims=True)
  return u / norm


# pylint: disable=too-many-locals
def _jsd_summary_loss(params, rng_key, apply_fn, **batch):
  y, theta = batch["y"], batch["theta"]
  n, p = theta.shape

  phi_key, rng_key = jr.split(rng_key)
  summr = apply_fn(params, method="summary", y=y)
  summr = jnp.tile(summr, [10, 1])
  theta = jnp.tile(theta, [10, 1])

  phi = _sample_unit_sphere(phi_key, 10, p)
  phi = jnp.repeat(phi, n, axis=0)

  second_summr = apply_fn(
    params, method="secondary_summary", y=summr, theta=phi
  )
  theta_prime = jnp.sum(theta * phi, axis=1).reshape(-1, 1)

  idx_pos = jnp.tile(jnp.arange(n), 10)
  perm_key, rng_key = jr.split(rng_key)
  idx_neg = jax.vmap(lambda x: jr.permutation(x, n))(
    jr.split(perm_key, 10)
  ).reshape(-1)
  f_pos = apply_fn(params, method="critic", y=second_summr, theta=theta_prime)
  f_neg = apply_fn(
    params,
    method="critic",
    y=second_summr[idx_pos],
    theta=theta_prime[idx_neg],
  )
  a, b = -jax.nn.softplus(-f_pos), jax.nn.softplus(f_neg)
  mi = a.mean() - b.mean()
  return -mi


# ruff: noqa: PLR0913
class NASSS(NASS):
  """Neural approximate slice sufficient statistics.

  Implements the NASSS algorithm introduced in :cite:t:`chen2021neural`.
  NASS can be used to automatically summary statistics of a data set.
  With the learned summaries, inferential algorithms like NLE or SMCABC
  can be used to infer posterior distributions.

  Args:
      model_fns: a tuple of calalbles. The first element needs to be a
          function that constructs a tfd.JointDistributionNamed, the second
          element is a simulator function.
      summary_net: a (neural) conditional density estimator
          to model the likelihood function of summary statistics, i.e.,
          the modelled dimensionality is that of the summaries
      summary_net: a SNASSSNet object

  Examples:
      >>> from jax import numpy as jnp
      >>> from sbijax import NASSS
      >>> from sbijax.nn import make_nasss_net
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      ...
      >>> prior = lambda: tfd.JointDistributionNamed(
      ...    dict(theta=tfd.Normal(jnp.zeros(5), 1.0))
      ... )
      >>> s = lambda seed, theta: tfd.Normal(
      ...     theta["theta"], 1.0).sample(seed=seed, sample_shape=(2,)
      ... ).reshape(-1, 10)
      >>> fns = prior, s
      >>> neural_network = make_nasss_net([64, 64, 5], [64, 64, 1], [64, 64, 1])
      >>> model = NASSS(fns, neural_network)

  References:
      Yanzhi Chen et al. "Is Learning Summary Statistics Necessary for Likelihood-free Inference". ICML, 2023
  """

  # pylint: disable=useless-parent-delegation
  def __init__(self, model_fns, summary_net):
    super().__init__(model_fns, summary_net)

  def _fit_summary_net(
    self,
    rng_key,
    train_iter,
    val_iter,
    optimizer,
    n_iter,
    n_early_stopping_patience,
  ):
    init_key, rng_key = jr.split(rng_key)
    params = self._init_summary_net_params(init_key, **next(iter(train_iter)))

    def loss_fn(params, rng, **batch):
      return _jsd_summary_loss(params, rng, self.model.apply, **batch)

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
    )
