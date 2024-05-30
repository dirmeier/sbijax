import haiku as hk
import jax
from jax import numpy as jnp


def make_mlp(
    n_layers: int = 2,
    hidden_size: int = 64,
    activation=jax.nn.gelu,
    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
    b_init=jnp.zeros,
):
    """Create a MLP-based classifier network.

    Args:
        n_layers: the number of hidden layers to be used
        hidden_size: the size of each layer
        activation: a JAX activation function
        w_init: a haiku initializer
        b_init: a haiku initializer

    Returns:
        a transformable haiku neural network module
    """

    @hk.without_apply_rng
    @hk.transform
    def _net(inputs, **kwargs):
        nn = hk.nets.MLP(
            output_sizes=[hidden_size] * n_layers + [1],
            w_init=w_init,
            b_init=b_init,
            activation=activation,
        )
        return nn(inputs)

    return _net
