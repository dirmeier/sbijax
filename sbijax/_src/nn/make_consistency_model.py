from collections.abc import Callable
from typing import Any

import haiku as hk
import jax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.stats import norm

from sbijax._src.nn.make_resnet import _ResnetBlock

__all__ = ["ConsistencyModel", "make_cm"]

# constants of the consistency training recipe used by :cite:t:`schmitt2023con`
# (their Appendix A): the curvature of the noise-level grid, the lognormal
# over noise levels that intervals are drawn from, and the pseudo-Huber scale
# per square-root dimension.
_RHO = 7.0
_P_MEAN = -1.1
_P_STD = 2.0
_HUBER_SCALE = 0.00054


# ruff: noqa: PLR0913,D417
class ConsistencyModel(hk.Module):
  """A consistency model.

  Args:
      n_dimension: the dimensionality of the modelled space
      transform: a haiku module. The transform is a callable that has to
          take as input arguments named 'theta', 'time', 'context' and
          **kwargs. Theta, time and context are two-dimensional arrays
          with the same batch dimensions.
      t_min: minimal time point for ODE integration
      t_max: maximal time point for ODE integration
      n_sampling_steps: number of consistency function evaluations when
          sampling
  """

  def __init__(
    self,
    n_dimension: int,
    transform: Callable[..., Any],
    t_min: float = 0.001,
    t_max: float = 200.0,
    n_sampling_steps: int = 10,
  ):
    """Construct a consistency model.

    Args:
        n_dimension: the dimensionality of the modelled space
        transform: a haiku module. The transform is a callable that has to
            take as input arguments named 'theta', 'time', 'context' and
            **kwargs. Theta, time and context are two-dimensional arrays
            with the same batch dimensions.
        t_min: minimal time point for ODE integration
        t_max: maximal time point for ODE integration
        n_sampling_steps: number of consistency function evaluations when
            sampling
    """
    super().__init__()
    self._n_dimension = n_dimension
    self._network = transform
    self._t_max = t_max
    self._t_min = t_min
    self._n_sampling_steps = n_sampling_steps

  def __call__(self, method, **kwargs):
    """Apply the consistency model.

    Args:
        method (str): method to call

    Keyword Args:
        keyword arguments for the called method:
    """
    return getattr(self, method)(**kwargs)

  def sample(self, context: jax.Array, **kwargs) -> jax.Array:
    """Sample with the multistep consistency sampler.

    Starts from ``N(0, t_max^2 I)`` noise and alternates evaluating the
    consistency function with re-noising on the descending time grid, as in
    Algorithm 1 of :cite:t:`schmitt2023con`.

    Args:
        context: array of conditioning variables
        kwargs: keyword arguments like 'is_training'

    Returns:
        an array of samples with one row per row of ``context``
    """
    shape = (context.shape[0], self._n_dimension)
    theta = self._t_max * jr.normal(hk.next_rng_key(), shape)
    theta = self.vector_field(theta, self._t_max, context, **kwargs)
    n_steps = self._n_sampling_steps
    times = self._time_grid(jnp.arange(n_steps + 1), n_steps)[::-1]
    for time in times[1:-1]:
      noise = jr.normal(hk.next_rng_key(), shape)
      theta_t = theta + jnp.sqrt(time**2 - self._t_min**2) * noise
      theta = self.vector_field(theta_t, time, context, **kwargs)
    return theta

  def loss(
    self,
    inputs: jax.Array,
    context: jax.Array,
    n_intervals: jax.Array,
    is_training: bool,
  ) -> jax.Array:
    """Compute the consistency training loss of :cite:t:`schmitt2023con`.

    The student evaluates the consistency function at the upper end of a grid
    interval and the teacher, a stop-gradient copy of the student, at the
    lower end. Both calls share one dropout random state.

    Args:
        inputs: array of parameters
        context: array of conditioning variables
        n_intervals: number of intervals of the time grid at this
            training step
        is_training: whether dropout is active

    Returns:
        an array of per-sample losses
    """
    index = self._sample_interval(
      hk.next_rng_key(), inputs.shape[0], n_intervals
    )
    time = self._time_grid(index, n_intervals).reshape(-1, 1)
    time_next = self._time_grid(index + 1, n_intervals).reshape(-1, 1)
    noise = jr.normal(hk.next_rng_key(), inputs.shape)

    dropout_key = hk.next_rng_key()
    with hk.with_rng(dropout_key):
      student = self.vector_field(
        inputs + time_next * noise, time_next, context, is_training=is_training
      )
    with hk.with_rng(dropout_key):
      teacher = self.vector_field(
        inputs + time * noise, time, context, is_training=is_training
      )
    teacher = jax.lax.stop_gradient(teacher)

    huber = _HUBER_SCALE * jnp.sqrt(self._n_dimension)
    squared_distance = jnp.sum(jnp.square(student - teacher), axis=-1)
    distance = jnp.sqrt(squared_distance + huber**2) - huber
    weight = 1.0 / (time_next - time).reshape(-1)
    return weight * distance

  def _time_grid(self, index, n_intervals):
    """Return the ``index``-th of ``n_intervals + 1`` grid time points."""
    left = self._t_min ** (1 / _RHO)
    right = self._t_max ** (1 / _RHO)
    return (left + index / n_intervals * (right - left)) ** _RHO

  def _sample_interval(self, rng_key, n_samples, n_intervals):
    """Draw grid intervals with probability proportional to lognormal mass.

    Draws a time from the lognormal truncated to ``[t_min, t_max]`` and
    returns the index of the grid interval containing it, which samples an
    interval with the probabilities of :cite:t:`schmitt2023con`.
    """
    lower = norm.cdf(jnp.log(self._t_min), _P_MEAN, _P_STD)
    upper = norm.cdf(jnp.log(self._t_max), _P_MEAN, _P_STD)
    quantile = jr.uniform(rng_key, (n_samples,), minval=lower, maxval=upper)
    time = jnp.exp(norm.ppf(quantile, _P_MEAN, _P_STD))
    left = self._t_min ** (1 / _RHO)
    right = self._t_max ** (1 / _RHO)
    position = n_intervals * (time ** (1 / _RHO) - left) / (right - left)
    index = jnp.clip(jnp.floor(position), 0, n_intervals - 1)
    return index.astype(jnp.int32)

  def vector_field(self, theta, time, context, **kwargs):
    """Compute the vector field.

    Args:
        theta: array of parameters
        time: time variables
        context: array of conditioning variables

    Keyword Args:
        keyword arguments that aer passed tothe neural network
    """
    time = jnp.full((theta.shape[0], 1), time)
    return self._network(theta=theta, time=time, context=context, **kwargs)


# pylint: disable=too-many-arguments,too-many-instance-attributes
class _CMResnet(hk.Module):
  """A simplified 1-d residual network."""

  def __init__(
    self,
    n_layers: int,
    n_dimension: int,
    hidden_size: int,
    activation: Callable[..., Any] = jax.nn.relu,
    dropout_rate: float = 0.0,
    t_min: float = 0.001,
    sigma_data: float = 1.0,
  ):
    super().__init__()
    self.n_layers = n_layers
    self.n_dimension = n_dimension
    self.hidden_size = hidden_size
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.sigma_data = sigma_data
    self.var_data = self.sigma_data**2
    self.t_min = t_min

  def __call__(self, theta, time, context, is_training, **kwargs):
    outputs = context
    t_theta_embedding = jnp.concatenate(
      [
        hk.Linear(self.n_dimension)(theta),
        hk.Linear(self.n_dimension)(time),
      ],
      axis=-1,
    )
    outputs = hk.Linear(self.hidden_size)(outputs)
    outputs = self.activation(outputs)
    for _ in range(self.n_layers):
      outputs = _ResnetBlock(
        hidden_size=self.hidden_size,
        activation=self.activation,
        dropout_rate=self.dropout_rate,
      )(outputs, context=t_theta_embedding, is_training=is_training)
    outputs = self.activation(outputs)
    outputs = hk.Linear(self.n_dimension)(outputs)

    # TODO(simon): dan we choose sigma automatically?
    out_skip = self._c_skip(time) * theta + self._c_out(time) * outputs
    return out_skip

  def _c_skip(self, time):
    return self.var_data / ((time - self.t_min) ** 2 + self.var_data)

  def _c_out(self, time):
    return (
      self.sigma_data * (time - self.t_min) / jnp.sqrt(self.var_data + time**2)
    )


# ruff: noqa: PLR0913
def make_cm(
  n_dimension: int,
  n_layers: int = 2,
  hidden_size: int = 64,
  activation: Callable[..., Any] = jax.nn.tanh,
  dropout_rate: float = 0.2,
  t_min: float = 0.001,
  t_max: float = 200.0,
  sigma_data: float = 1.0,
  n_sampling_steps: int = 10,
):
  """Create a consistency model.

  The consistency model uses a residual network as score network.

  Args:
      n_dimension: dimensionality of modelled space
      n_layers: number of resnet blocks
      hidden_size: sizes of hidden layers for each resnet block
      activation: a jax activation function
      dropout_rate: dropout rate to use in resnet blocks
      t_min: minimal time point for ODE integration
      t_max: maximal time point for ODE integration
      sigma_data: the standard deviation of the data :)
      n_sampling_steps: number of consistency function evaluations when
          sampling; :cite:t:`schmitt2023con` recommend 5 to 15

  Returns:
      a consistency model
  """

  @hk.transform
  def _cm(method, **kwargs):
    nn = _CMResnet(
      n_layers=n_layers,
      n_dimension=n_dimension,
      hidden_size=hidden_size,
      activation=activation,
      dropout_rate=dropout_rate,
      t_min=t_min,
      sigma_data=sigma_data,
    )
    cm = ConsistencyModel(
      n_dimension,
      nn,
      t_min=t_min,
      t_max=t_max,
      n_sampling_steps=n_sampling_steps,
    )
    return cm(method, **kwargs)

  return _cm
