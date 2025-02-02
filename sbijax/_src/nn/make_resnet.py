from typing import Callable

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
        activation: Callable = jax.nn.relu,
        dropout_rate: float = 0.2,
        do_batch_norm: bool = False,
        batch_norm_decay: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = activation
        self.do_batch_norm = do_batch_norm
        self.dropout_rate = dropout_rate
        self.batch_norm_decay = batch_norm_decay

    def __call__(self, inputs, context=None, is_training=False):
        outputs = inputs
        if self.do_batch_norm:
            outputs = hk.BatchNorm(True, True, self.batch_norm_decay)(
                outputs, is_training=is_training
            )
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
        activation: Callable = jax.nn.relu,
        dropout_rate: float = 0.1,
        do_batch_norm: bool = True,
        batch_norm_decay: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.do_batch_norm = do_batch_norm
        self.dropout_rate = dropout_rate
        self.batch_norm_decay = batch_norm_decay

    def __call__(self, inputs, is_training=False, **kwargs):
        outputs = inputs
        outputs = hk.Linear(self.hidden_size)(outputs)
        outputs = self.activation(outputs)
        for _ in range(self.n_layers):
            outputs = _ResnetBlock(
                hidden_size=self.hidden_size,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                do_batch_norm=self.do_batch_norm,
                batch_norm_decay=self.batch_norm_decay,
            )(outputs, is_training=is_training)
        outputs = self.activation(outputs)
        outputs = hk.Linear(1)(outputs)
        return outputs


def make_resnet(
    n_layers: int = 2,
    hidden_size: int = 64,
    activation: Callable = jax.nn.tanh,
    dropout_rate: float = 0.2,
    do_batch_norm: bool = False,
    batch_norm_decay: float = 0.2,
):
    """Create a ResNet-based classifier network.

    Args:
        n_layers: number of normalizing flow layers
        hidden_size: sizes of hidden layers for each normalizing flow
        activation: a jax activation function
        dropout_rate: dropout rate to use in resnet blocks
        do_batch_norm: use batch normalization or not
        batch_norm_decay: decay rate of EMA in batch norm layer
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
            do_batch_norm=do_batch_norm,
            dropout_rate=dropout_rate,
            batch_norm_decay=batch_norm_decay,
        )
        return nn(inputs, is_training=is_training)

    return _net
