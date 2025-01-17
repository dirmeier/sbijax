from collections.abc import Iterable
from typing import Callable

import haiku as hk
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


# pylint: disable=too-many-arguments
def make_mdn(
    n_dimension: int,
    n_components: int,
    hidden_sizes: Iterable[int] = (64, 64),
    activation: Callable = jax.nn.relu,
):
    """Create a mixture density network.

    The MDN uses `n_components` mixture components each modelling the distribution of
    a `n_dimension`al data point.

    Args:
        n_dimension: dimensionality of data
        n_components: number of mixture components
        hidden_sizes: sizes of hidden layers for each normalizing flow. E.g.,
            when the hidden sizes are a tuple (64, 64), then each maf layer
            uses a MADE with two layers of size 64 each
        activation: a jax activation function

    Returns:
        a mixture density network
    """

    @hk.transform
    def mdn(method, **kwargs):
        n = kwargs["x"].shape[0]
        hidden = hk.nets.MLP(
            hidden_sizes, activation=activation, activate_final=True
        )(kwargs["x"])
        logits = hk.Linear(n_components)(hidden)
        mu_sigma = hk.Linear(n_components * n_dimension * 2)(hidden)
        mu, sigma = jnp.split(mu_sigma, 2, axis=-1)

        mixture = tfd.MixtureSameFamily(
            tfd.Categorical(logits=logits),
            tfd.MultivariateNormalDiag(
                mu.reshape(n, n_components, n_dimension),
                jnp.exp(sigma.reshape(n, n_components, n_dimension)),
            ),
        )
        if method == "sample":
            return mixture.sample(seed=hk.next_rng_key())
        else:
            return mixture.log_prob(kwargs["y"])

    return mdn


make_mdn(2, 2)
