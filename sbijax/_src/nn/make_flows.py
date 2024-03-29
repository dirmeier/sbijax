from collections.abc import Iterable
from typing import Callable

import distrax
import haiku as hk
import jax
from jax import numpy as jnp
from surjectors import (
    AffineMaskedAutoregressiveInferenceFunnel,
    Chain,
    MaskedAutoregressive,
    Permutation,
    TransformedDistribution,
)
from surjectors.nn import MADE, make_mlp
from surjectors.util import unstack


def _bijector_fn(params):
    means, log_scales = unstack(params, -1)
    return distrax.ScalarAffine(means, jnp.exp(log_scales))


def _decoder_fn(n_dim, hidden_size):
    decoder_net = make_mlp(
        hidden_size + [n_dim * 2],
        w_init=hk.initializers.TruncatedNormal(stddev=0.001),
    )

    def _fn(z):
        params = decoder_net(z)
        mu, log_scale = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)), 1)

    return _fn


# pylint: disable=too-many-arguments
def make_affine_maf(
    n_dimension: int,
    n_layers: int = 5,
    hidden_sizes: Iterable[int] = (64, 64),
    activation: Callable = jax.nn.tanh,
):
    """Create an affine masked autoregressive flow.

    The MAFs use `n_layers` layers and are parameterized using MADE networks
    with `hidden_sizes` neurons per layer.

    Args:
        n_dimension: dimensionality of data
        n_layers: number of normalizing flow layers
        hidden_sizes: sizes of hidden layers for each normalizing flow. E.g.,
            when the hidden sizes are a tuple (64, 64), then each maf layer
            uses a MADE with two layers of size 64 each
        activation: a jax activation function

    Returns:
        a normalizing flow model
    """

    @hk.without_apply_rng
    @hk.transform
    def _flow(method, **kwargs):
        layers = []
        order = jnp.arange(n_dimension)
        for _ in range(n_layers):
            layer = MaskedAutoregressive(
                bijector_fn=_bijector_fn,
                conditioner=MADE(
                    n_dimension,
                    list(hidden_sizes) + [n_dimension * 2],
                    2,
                    w_init=hk.initializers.TruncatedNormal(0.001),
                    b_init=jnp.zeros,
                    activation=activation,
                ),
            )
            order = order[::-1]
            layers.append(layer)
            layers.append(Permutation(order, 1))
        chain = Chain(layers[:-1])

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dimension), jnp.ones(n_dimension)),
            1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    return _flow


def make_surjective_affine_maf(
    n_dimension: int,
    n_layer_dimensions: Iterable[int],
    hidden_sizes: Iterable[int] = (64, 64),
    activation: Callable = jax.nn.tanh,
):
    """Create a surjective affine masked autoregressive flow.

    The MAFs use `n_layers` layers and are parameterized using MADE networks
    with `hidden_sizes` neurons per layer. For each dimensionality reducing
    layer, a conditional Gaussian density is used that uses the same number of
    layer and nodes per layers as `hidden_sizes`. The argument
    `n_layer_dimensions` determines which layer is dimensionality-preserving
    or -reducing. For example, for `n_layer_dimensions=(5, 5, 3, 3)` and
    `n_dimension=5`, the third layer would reduce the dimensionality by two
    and use a surjection layer. THe other layers are dimensionality-preserving.

    Args:
        n_dimension: a list of integers that determine the dimensionality
            of each flow layer
        n_layer_dimensions: list of integers that determine if a layer is
            dimensionality-preserving or -reducing
        hidden_sizes: sizes of hidden layers for each normalizing flow
        activation: a jax activation function

    Examples:
        >>> make_surjective_affine_maf(10, (10, 10, 5, 5, 5))

    Returns:
        a surjective normalizing flow model
    """

    @hk.without_apply_rng
    @hk.transform
    def _flow(method, **kwargs):
        layers = []
        order = jnp.arange(n_dimension)
        curr_dim = n_dimension
        for i, n_dim_curr_layer in enumerate(n_layer_dimensions):
            # layer is dimensionality preserving
            if n_dim_curr_layer == curr_dim:
                layer = MaskedAutoregressive(
                    bijector_fn=_bijector_fn,
                    conditioner=MADE(
                        n_dim_curr_layer,
                        list(hidden_sizes) + [n_dim_curr_layer * 2],
                        2,
                        w_init=hk.initializers.TruncatedNormal(0.001),
                        b_init=jnp.zeros,
                        activation=activation,
                    ),
                )
                order = order[::-1]
            elif n_dim_curr_layer < curr_dim:
                n_latent = n_dim_curr_layer
                layer = AffineMaskedAutoregressiveInferenceFunnel(
                    n_latent,
                    _decoder_fn(curr_dim - n_latent, list(hidden_sizes)),
                    conditioner=MADE(
                        n_latent,
                        list(hidden_sizes) + [n_dim_curr_layer * 2],
                        2,
                        w_init=hk.initializers.TruncatedNormal(0.001),
                        b_init=jnp.zeros,
                        activation=jax.nn.tanh,
                    ),
                )
                curr_dim = n_latent
                order = order[::-1]
                order = order[:curr_dim] - jnp.min(order[:curr_dim])
            else:
                raise ValueError(
                    f"n_dimension at layer {i} is layer than the dimension of"
                    f" the following layer {i + 1}"
                )
            layers.append(layer)
            layers.append(Permutation(order, 1))
        chain = Chain(layers[:-1])

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dimension), jnp.ones(n_dimension)),
            1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    return _flow
