from collections.abc import Iterable
from typing import Callable

import haiku as hk
import jax

from sbijax._src.nn.nass_net import NASSNet
from sbijax._src.nn.nasss_net import NASSSNet


def make_nass_net(
    embedding_dim: int,
    hidden_sizes: Iterable[int],
    activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
):
    """Create a critic network for SNASS.

    Args:
        embedding_dim: dimensionality of the summary statistic
        hidden_sizes: list of integers specifying the hidden dimensions
            of the networks
        activation: a jax activation function

    Returns:
        a network that can be used within a NASS posterior estimator
    """

    @hk.without_apply_rng
    @hk.transform
    def _net(method, **kwargs):
        summary_net = hk.nets.MLP(
            output_sizes=list(hidden_sizes) + [embedding_dim],
            activation=activation,
        )
        critic_net = hk.nets.MLP(
            output_sizes=list(hidden_sizes) + [1], activation=activation
        )
        net = NASSNet(summary_net=summary_net, critic_net=critic_net)
        return net(method, **kwargs)

    return _net


def make_nasss_net(
    embedding_dim: int,
    sec_embedding_dim: int,
    hidden_sizes: Iterable[int],
    activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
):
    """Create a critic network for SNASSS.

    Args:
        embedding_dim: dimensionality of the summary statistic
        sec_embedding_dim: dimensionality of the secondary
            summary statistic
        hidden_sizes: list of integers specifying the hidden dimensions
            of the networks
        activation: a jax activation function

    Returns:
        a network that can be used within a SNASSS posterior estimator
    """

    @hk.without_apply_rng
    @hk.transform
    def _net(method, **kwargs):
        summary_net = hk.nets.MLP(
            output_sizes=list(hidden_sizes) + [embedding_dim],
            activation=activation,
        )
        sec_summary_net = hk.nets.MLP(
            output_sizes=list(hidden_sizes) + [sec_embedding_dim],
            activation=activation,
        )
        critic_net = hk.nets.MLP(
            output_sizes=list(hidden_sizes) + [1], activation=activation
        )
        net = NASSSNet(
            summary_net=summary_net,
            sec_summary_net=sec_summary_net,
            critic_net=critic_net,
        )
        return net(method, **kwargs)

    return _net
