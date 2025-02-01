import dataclasses
from typing import Callable, Optional

import haiku as hk
import jax
from einops import rearrange
from jax import numpy as jnp

__all__ = ["make_simformer_based_score_model"]

from sbijax._src.experimental.nn.make_score_network import (
    ScoreModel,
    timestep_embedding,
)


@dataclasses.dataclass
class _Encoder(hk.Module):
    num_heads: int
    num_layers: int
    head_size: int
    dropout_rate: float
    widening_factor: int = 4
    initializer: Callable = hk.initializers.TruncatedNormal(stddev=0.01)
    activation: Callable = jax.nn.gelu

    def __call__(self, inputs, time, mask, *, is_training):
        dropout_rate = self.dropout_rate if is_training else 0.0
        mask = mask[None, None, ...] if mask is not None else None
        hidden = inputs
        for _ in range(self.num_layers):
            intr = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
                hidden
            )
            intr = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.head_size or (intr.shape[-1] // self.num_heads),
                w_init=self.initializer,
            )(intr, intr, intr, mask=mask)
            intr = hk.dropout(hk.next_rng_key(), dropout_rate, intr)
            hidden = hidden + intr

            intr = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
                hidden
            )
            intr = hk.nets.MLP(
                [self.widening_factor * intr.shape[-1], intr.shape[-1]],
                w_init=self.initializer,
                activation=jax.nn.gelu,
            )(intr)
            intr = hk.dropout(hk.next_rng_key(), dropout_rate, intr)
            hidden = hidden + intr

        hidden = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            hidden
        )
        return hidden


@dataclasses.dataclass
class _SimFormer(hk.Module):
    mask: jax.Array
    n_heads: int = 4
    n_layers: int = 4
    head_size: Optional[int] = None
    embedding_dim_values: int = 32
    embedding_dim_ids: int = 32
    embedding_dim_conditioning: int = 10
    time_embedding_layers: tuple[int, ...] = (128, 128)
    dropout_rate: float = 0.1
    activation: Callable = jax.nn.relu

    def __call__(self, inputs, time, context, *, is_training=True):
        n_inputs, n_context = inputs.shape[-1], context.shape[-1]
        inputs = jnp.concatenate([inputs, context], axis=-1)

        time = hk.Sequential(
            [
                lambda x: timestep_embedding(x, self.time_embedding_layers[0]),
                hk.nets.MLP(
                    self.time_embedding_layers, activation=self.activation
                ),
            ]
        )(time)

        ids = jnp.arange(inputs.shape[-1], dtype=jnp.int32).reshape(1, -1)
        condition_mask = jnp.concatenate(
            [
                jnp.ones(n_inputs, dtype=jnp.int32),
                jnp.zeros(n_context, dtype=jnp.int32),
            ]
        ).reshape(1, -1)
        ids, condition_mask, inputs = jnp.broadcast_arrays(
            ids, condition_mask, inputs
        )
        inputs_embedding = jnp.tile(
            inputs.reshape(*inputs.shape, 1), [1, 1, self.embedding_dim_values]
        )
        id_embedding = hk.Embed(inputs.shape[-1], self.embedding_dim_ids)(ids)
        condition_mask_embedding = hk.Embed(2, self.embedding_dim_conditioning)(
            condition_mask
        )
        inputs = jnp.concatenate(
            [inputs_embedding, id_embedding, condition_mask_embedding], axis=-1
        )
        hidden = _Encoder(
            num_heads=self.n_heads,
            num_layers=self.n_layers,
            head_size=self.head_size,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        )(inputs, time, self.mask, is_training=is_training)
        hidden = hk.Linear(1)(hidden)
        outputs = rearrange(hidden, "b l d -> b (l d)")
        outputs = outputs[..., :n_inputs]
        return outputs


# ruff: noqa: PLR0913
def make_simformer_based_score_model(
    n_dimension: int,
    mask: jax.Array,
    n_heads=4,
    n_layers=4,
    head_size: Optional[int] = None,
    embedding_dim_values=32,
    embedding_dim_ids=32,
    embedding_dim_conditioning=8,
    time_embedding_layers=(
        128,
        128,
    ),
    dropout_rate=0.1,
    activation=jax.nn.gelu,
    sde="vp",
    beta_min=0.1,
    beta_max=10.0,
    time_eps=0.001,
    time_max=1,
):
    """Create a score network for AiO.

    The score model uses a transformer a score estimator.

    Args:
        n_dimension: dimensionality of modelled space
        mask: a binary matrix of conditional dependencies
        n_heads: number of attention heads
        n_layers: number of attention layers
        head_size: size of an attention head
        embedding_dim_values: dimensionality of the embedding for the values
        embedding_dim_ids: dimensionality of the embedding for the ids
            of the variables
        embedding_dim_conditioning: dimensionality of the binary
            conditioning labels
        time_embedding_layers: a tuple if ints determining the output sizes of
            the data embedding network
        dropout_rate: a tuple if ints determining the output sizes of
            the data embedding network
        activation: activation function to be used for
        sde: can be either of 'vp' and 've'. Defines the type of SDE to be used
            as a forward process. See the original publication and references
            therein for details.
        beta_min: beta min. Again, see the paper please.
        beta_max: beta max. Again, see the paper please.
        time_eps: some small number to use as minimum time point for the
            forward process. Used for numerical stability.
        time_max: maximum integration time. 1 is good, but so is 5 or 10.

    Returns:
        returns a conditional continuous normalizing flow
    """

    @hk.transform
    def _score_model(method, **kwargs):
        nn = _SimFormer(
            mask=mask,
            n_heads=n_heads,
            n_layers=n_layers,
            head_size=head_size,
            embedding_dim_conditioning=embedding_dim_conditioning,
            embedding_dim_values=embedding_dim_values,
            embedding_dim_ids=embedding_dim_ids,
            time_embedding_layers=time_embedding_layers,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        net = ScoreModel(
            n_dimension, nn, sde, beta_min, beta_max, time_eps, time_max
        )
        return net(method, **kwargs)

    return _score_model
