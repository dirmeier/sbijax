import chex
import jax.tree
from jax import random as jr


def test_correct_prior_dims(simulator_model):
    prior, *_ = simulator_model()
    for shape, ndim, axis in zip(
      [(), (1,), (10,)],
      [1, 2, 2],
      [None, 0, 0]
    ):
      theta = prior.sample(seed=jr.key(1), sample_shape=shape)
      jax.tree.map(lambda x: chex.assert_equal(x.ndim, ndim), theta)
      if axis is not None:
        jax.tree.map(lambda x: chex.assert_axis_dimension(x, axis, shape[0]), theta)


def test_correct_simulator_dims(simulator_model):
    prior, simulator, _ = simulator_model()
    for shape, n_obs in zip([(), (1,), (10,)], [1, 1, 10]):
      theta = prior.sample(seed=jr.key(1), sample_shape=shape)
      sim = simulator(jr.key(0), theta)
      chex.assert_rank(sim, 2)
      chex.assert_axis_dimension(sim, 0, n_obs)


def test_correct_likleihood_dims(simulator_model):
    prior, simulator, lik_fn = simulator_model()
    if lik_fn is None:
      return
    for shape, n_obs in zip([(), (1,), (10,)], [1, 1, 10]):
      theta = prior.sample(seed=jr.key(1), sample_shape=shape)
      sim = simulator(jr.key(0), theta)
      log_lik = lik_fn(sim, theta)
      chex.assert_rank(log_lik, 1)
      chex.assert_axis_dimension(sim, 0, n_obs)
