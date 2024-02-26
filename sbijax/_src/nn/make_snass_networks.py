from typing import Callable, Iterable

import haiku as hk
import jax

from sbijax._src.nn.snass_net import SNASSNet
from sbijax._src.nn.snasss_net import SNASSSNet


def make_snass_net(
    summary_net_dimensions: Iterable[int],
    critic_net_dimensions: Iterable[int],
    activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
):
    """Create a critic network for SNASS.

    Args:
        summary_net_dimensions: a list of integers representing
            the dimensionalities of the summary network. The _last_ dimension
            determines the dimensionality of the summary statistic
        critic_net_dimensions: a list of integers representing the
            dimensionality of the critic network. The _last_ dimension
            needs to be 1.
        activation: a jax activation function

    Returns:
        a network that can be used within a SNASS posterior estimator
    """

    @hk.without_apply_rng
    @hk.transform
    def _net(method, **kwargs):
        summary_net = hk.nets.MLP(
            output_sizes=summary_net_dimensions, activation=activation
        )
        critic_net = hk.nets.MLP(
            output_sizes=critic_net_dimensions, activation=activation
        )
        net = SNASSNet(summary_net=summary_net, critic_net=critic_net)
        return net(method, **kwargs)

    return _net


def make_snasss_net(
    summary_net_dimensions: Iterable[int],
    sec_summary_net_dimensions: Iterable[int],
    critic_net_dimensions: Iterable[int],
    activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
):
    """Create a critic network for SNASSS.

    Args:
        summary_net_dimensions: a list of integers representing
            the dimensionalities of the summary network. The _last_ dimension
            determines the dimensionality of the summary statistic
        sec_summary_net_dimensions:  list of integers representing
            the dimensionalities of the summary network. The _last_ dimension
            determines the dimensionality of the second summary statistic and
            it should be smaller than the last dimension of the
            first summary net.
        critic_net_dimensions: a list of integers representing the
            dimensionality of the critic network. The _last_ dimension
            needs to be 1.
        activation: a jax activation function

    Returns:
        a network that can be used within a SNASSS posterior estimator
    """

    @hk.without_apply_rng
    @hk.transform
    def _net(method, **kwargs):
        summary_net = hk.nets.MLP(
            output_sizes=summary_net_dimensions, activation=activation
        )
        sec_summary_net = hk.nets.MLP(
            output_sizes=sec_summary_net_dimensions, activation=activation
        )
        critic_net = hk.nets.MLP(
            output_sizes=critic_net_dimensions, activation=activation
        )
        net = SNASSSNet(
            summary_net=summary_net,
            sec_summary_net=sec_summary_net,
            critic_net=critic_net,
        )
        return net(method, **kwargs)

    return _net
