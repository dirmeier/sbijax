from collections.abc import Callable
from typing import Any

import haiku as hk
import jax
from jax import numpy as jnp
from jax._src.nn.functions import glu


# pylint: disable=too-many-arguments
class _ResnetBlock(hk.Module):
  """A block for a 1d residual network."""

  def __init__(
    self,
    hidden_size: int,
    activation: Callable[..., Any] = jax.nn.relu,
    dropout_rate: float = 0.2,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.activation = activation
    self.dropout_rate = dropout_rate

  def __call__(self, inputs, context=None, is_training=False):
    outputs = inputs
    outputs = hk.Linear(self.hidden_size)(outputs)
    outputs = self.activation(outputs)
    if is_training:
      outputs = hk.dropout(
        rng=hk.next_rng_key(), rate=self.dropout_rate, x=outputs
      )
    outputs = hk.Linear(self.hidden_size)(outputs)
    if context is not None:
      context_proj = hk.Linear(inputs.shape[-1])(context)
      outputs = glu(jnp.concatenate([outputs, context_proj], axis=-1))
    return outputs + inputs


# ruff: noqa: PLR0913
class _Resnet(hk.Module):
  """A simplified 1-d residual network."""

  def __init__(
    self,
    n_layers: int,
    hidden_size: int,
    activation: Callable[..., Any] = jax.nn.relu,
    dropout_rate: float = 0.1,
  ):
    super().__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.activation = activation
    self.dropout_rate = dropout_rate

  def __call__(self, inputs, is_training=False, **kwargs):
    outputs = inputs
    outputs = hk.Linear(self.hidden_size)(outputs)
    outputs = self.activation(outputs)
    for _ in range(self.n_layers):
      outputs = _ResnetBlock(
        hidden_size=self.hidden_size,
        activation=self.activation,
        dropout_rate=self.dropout_rate,
      )(outputs, is_training=is_training)
    outputs = self.activation(outputs)
    outputs = hk.Linear(1)(outputs)
    return outputs


def make_resnet(
  n_layers: int = 2,
  hidden_size: int = 64,
  activation: Callable[..., Any] = jax.nn.tanh,
  dropout_rate: float = 0.2,
):
  """Create a ResNet-based classifier network.

  Args:
      n_layers: number of normalizing flow layers
      hidden_size: sizes of hidden layers for each normalizing flow
      activation: a jax activation function
      dropout_rate: dropout rate to use in resnet blocks
  Returns:
      a neural network model
  """

  @hk.without_apply_rng
  @hk.transform
  def _net(inputs, is_training=False):
    nn = _Resnet(
      n_layers=n_layers,
      hidden_size=hidden_size,
      activation=activation,
      dropout_rate=dropout_rate,
    )
    return nn(inputs, is_training=is_training)

  return _net
