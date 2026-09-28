import dataclasses
from collections.abc import Callable
from typing import Any

import haiku as hk
import jax
from einops import rearrange
from jax import numpy as jnp

__all__ = ["make_simformer_based_score_model"]

from sbijax._src.experimental.nn.make_score_network import (
  ScoreModel,
  noise_range,
  timestep_embedding,
)


@dataclasses.dataclass
class _Encoder(hk.Module):
  num_heads: int
  num_layers: int
  head_size: int | None
  dropout_rate: float
  widening_factor: int = 4
  initializer: Callable[..., Any] = hk.initializers.TruncatedNormal(stddev=0.01)
  activation: Callable[..., Any] = jax.nn.gelu

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
        model_size=intr.shape[-1],
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
        activation=self.activation,
      )(intr)
      intr = hk.dropout(hk.next_rng_key(), dropout_rate, intr)
      # every token receives the diffusion time in every layer, as in the
      # reference Simformer implementation
      time_embedding = self.activation(
        hk.Linear(intr.shape[-1], w_init=self.initializer)(time)
      )
      hidden = hidden + intr + time_embedding[:, None, :]

    hidden = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
      hidden
    )
    return hidden


@dataclasses.dataclass
class _SimFormer(hk.Module):
  mask: jax.Array
  n_heads: int = 4
  n_layers: int = 4
  head_size: int | None = None
  embedding_dim_values: int = 32
  embedding_dim_ids: int = 32
  embedding_dim_conditioning: int = 10
  time_embedding_layers: tuple[int, ...] = (128, 128)
  dropout_rate: float = 0.1
  activation: Callable[..., Any] = jax.nn.relu

  def __call__(self, inputs, time, context, *, is_training=True):
    n_inputs, n_context = inputs.shape[-1], context.shape[-1]
    inputs = jnp.concatenate([inputs, context], axis=-1)

    time = hk.Sequential(
      [
        lambda x: timestep_embedding(x, self.time_embedding_layers[0]),
        hk.nets.MLP(self.time_embedding_layers, activation=self.activation),
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
  n_heads: int = 4,
  n_layers: int = 4,
  head_size: int | None = None,
  embedding_dim_values: int = 32,
  embedding_dim_ids: int = 32,
  embedding_dim_conditioning: int = 8,
  time_embedding_layers: tuple[int, ...] = (
    128,
    128,
  ),
  dropout_rate: float = 0.1,
  activation: Callable[..., Any] = jax.nn.gelu,
  sde: str = "ve",
  beta_min: float = 0.1,
  beta_max: float = 10.0,
  sigma_min: float = 1e-4,
  sigma_max: float = 15.0,
  time_eps: float = 0.001,
  time_max: float = 1.0,
  scale_by_sigma: bool = False,
):
  """Create a score network for AiO.

  The score model uses a transformer as a score estimator.

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
          the time embedding network
      dropout_rate: dropout rate of the attention and MLP blocks
      activation: activation function of the time embedding and the MLP
          blocks
      sde: either of 'vp' and 've', the forward process. The VE noise
          scales follow the reference implementation of
          :cite:t:`gloeckler2024allinone`
      beta_min: minimal noise rate of the VP SDE
      beta_max: maximal noise rate of the VP SDE
      sigma_min: minimal noise scale of the VE SDE
      sigma_max: maximal noise scale of the VE SDE
      time_eps: some small number to use as minimum time point for the
          forward process. Used for numerical stability.
      time_max: maximum integration time
      scale_by_sigma: if true, divide the network output by the marginal std
          of the forward process, so that the network predicts the negative
          noise, as in the reference implementation of
          :cite:t:`gloeckler2024allinone`

  Returns:
      returns a score model that can be used for posterior inference using
      AiO.

  References:
      Gloeckler, Manuel, et al. "All-in-one simulation-based inference." International Conference on Machine Learning, 2024.
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
    noise_min, noise_max = noise_range(
      sde, beta_min, beta_max, sigma_min, sigma_max
    )
    net = ScoreModel(
      n_dimension,
      nn,
      sde,
      noise_min,
      noise_max,
      time_eps,
      time_max,
      scale_by_sigma=scale_by_sigma,
    )
    return net(method, **kwargs)

  return _score_model
