from collections.abc import Iterable
from typing import Callable, Optional

import distrax
import haiku as hk
import jax
import surjectors
from jax import numpy as jnp
from surjectors import (
    AffineMaskedAutoregressiveInferenceFunnel,
    Chain,
    MaskedAutoregressive,
    MaskedCoupling,
    MaskedCouplingInferenceFunnel,
    Permutation,
    TransformedDistribution,
)
from surjectors.nn import MADE
from surjectors.nn import make_mlp as surjectors_mlp
from surjectors.util import make_alternating_binary_mask, unstack
from tensorflow_probability.substrates.jax import distributions as tfd


# ruff: noqa: PLR0913, E501
def make_maf(
    n_dimension: int,
    n_layers: Optional[int] = 5,
    n_layer_dimensions: Optional[Iterable[int]] = None,
    hidden_sizes: Iterable[int] = (64, 64),
    activation: Callable = jax.nn.tanh,
) -> hk.Transformed:
    """Create an affine (surjective) masked autoregressive flow.

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
        n_layers: number of layers
        n_layer_dimensions: list of integers that determine if a layer is
            dimensionality-preserving or -reducing
        hidden_sizes: sizes of hidden layers for each normalizing flow
        activation: a jax activation function

    Examples:
        >>> neural_network = make_maf(10, n_layer_dimensions=(10, 10, 5, 5, 5))

    Returns:
        a (surjective) normalizing flow model
    """
    if isinstance(n_layers, int) and n_layer_dimensions is not None:
        assert n_layers == len(list(n_layer_dimensions))
    elif isinstance(n_layers, int):
        n_layer_dimensions = [n_dimension] * n_layers

    return _make_maf(
        n_dimension=n_dimension,
        n_layer_dimensions=n_layer_dimensions,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )


def _make_maf(
    n_dimension,
    n_layer_dimensions,
    hidden_sizes,
    activation,
):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return surjectors.ScalarAffine(means, jnp.exp(log_scales))

    def _decoder_fn(n_dim, hidden_sizes):
        decoder_net = surjectors_mlp(
            hidden_sizes + [n_dim * 2],
            w_init=hk.initializers.TruncatedNormal(stddev=0.001),
        )

        def _fn(z):
            params = decoder_net(z)
            mu, log_scale = jnp.split(params, 2, -1)
            return tfd.Independent(tfd.Normal(mu, jnp.exp(log_scale)), 1)

        return _fn

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
                        list(hidden_sizes),
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
                        list(hidden_sizes),
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

        base_distribution = tfd.Independent(
            tfd.Normal(jnp.zeros(curr_dim), jnp.ones(curr_dim)),
            1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    return _flow


# ruff: noqa: PLR0913, E501
def make_spf(
    n_dimension: int,
    range_min: float,
    range_max: float,
    n_layers: Optional[int] = 5,
    n_layer_dimensions: Optional[Iterable[int]] = None,
    hidden_sizes: Iterable[int] = (64, 64),
    n_params: int = 10,
    activation: Callable = jax.nn.tanh,
) -> hk.Transformed:
    """Create a rational-quadratic (surjective) spline coupling flow.

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
        range_min: minimum range on which the spline is defined
        range_max: maximum range on which the spline is defined
        n_layers: number of layers
        n_layer_dimensions: list of integers that determine if a layer is
            dimensionality-preserving or -reducing
        hidden_sizes: sizes of hidden layers for each normalizing flow
        n_params: number of parameters of each spline
        activation: a jax activation function

    Examples:
        >>> neural_network = make_spf(10, -1.0, 1.0, n_layer_dimensions=(10, 10, 5, 5, 5))

    Returns:
        a (surjective) normalizing flow model
    """
    if isinstance(n_layers, int) and n_layer_dimensions is not None:
        assert n_layers == len(list(n_layer_dimensions))
    if isinstance(n_layers, int):
        n_layer_dimensions = [n_dimension] * n_layers

    return _make_spf(
        n_dimension=n_dimension,
        range_min=range_min,
        range_max=range_max,
        n_layer_dimensions=n_layer_dimensions,
        hidden_sizes=hidden_sizes,
        n_params=n_params,
        activation=activation,
    )


def _make_spf(
    n_dimension,
    n_layer_dimensions,
    range_min,
    range_max,
    n_params,
    hidden_sizes,
    activation,
):
    def _bijector_fn(params):
        return distrax.RationalQuadraticSpline(
            params, range_min=range_min, range_max=range_max
        )

    def _decoder_fn(dims):
        def fn(z):
            params = surjectors_mlp(dims, activation=activation)(z)
            mu, log_scale = jnp.split(params, 2, -1)
            return tfd.Independent(tfd.Normal(mu, jnp.exp(log_scale)))

        return fn

    def _conditioner(n_dim):
        return hk.Sequential(
            [
                surjectors_mlp(
                    list(hidden_sizes) + [n_params * n_dim],
                    activation=activation,
                ),
                hk.Reshape((n_dimension, n_params)),
            ]
        )

    @hk.transform
    def _flow(method, **kwargs):
        layers = []
        curr_dim = n_dimension
        for i, n_dim_curr_layer in enumerate(n_layer_dimensions):
            # layer is dimensionality preserving
            if n_dim_curr_layer == curr_dim:
                layer = MaskedCoupling(
                    mask=make_alternating_binary_mask(curr_dim, i % 2 == 0),
                    conditioner=_conditioner(curr_dim),
                    bijector_fn=_bijector_fn,
                )
            # layer is dimensionality reducing
            elif n_dim_curr_layer < curr_dim:
                n_latent = n_dim_curr_layer
                layer = MaskedCouplingInferenceFunnel(
                    n_keep=n_latent,
                    decoder=_decoder_fn(
                        list(hidden_sizes) + [2 * (curr_dim - n_latent)]
                    ),
                    conditioner=surjectors_mlp(
                        list(hidden_sizes) + [2 * curr_dim],
                        activation=activation,
                    ),
                    bijector_fn=_bijector_fn,
                )
                curr_dim = n_latent
            else:
                raise ValueError(
                    f"n_dimension at layer {i} is layer than the dimension of"
                    f" the following layer {i + 1}"
                )
            layers.append(layer)
        chain = Chain(layers)

        base_distribution = tfd.Independent(
            tfd.Normal(jnp.zeros(n_dimension), jnp.ones(n_dimension)),
            1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    return _flow
