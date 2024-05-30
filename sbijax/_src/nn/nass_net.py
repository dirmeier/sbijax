from collections.abc import Iterable
from typing import Callable

import haiku as hk
import jax
from jax import numpy as jnp


# ruff: noqa: PLR0913,S101
class NASSNet(hk.Module):
    """A network for NASS."""

    def __init__(
        self,
        summary_net_dimensions: Iterable[int] = None,
        critic_net_dimensions: Iterable[int] = None,
        summary_net: Callable = None,
        critic_net: Callable = None,
    ):
        """Constructs a NASSNet.

        Can be used either by providing network dimensions or haiku modules.

        Args:
            summary_net_dimensions: a list of integers representing
                the dimensionalities of the summary network. The _last_
                dimension determines the dimensionality of the summary statistic
            critic_net_dimensions: a list of integers representing the
                dimensionality of the critic network. The _last_ dimension
                needs to be 1.
            summary_net: a haiku MLP with trailing dimension being the
                dimensionality of the summary statistic
            critic_net: : a haiku MLP with a trailing dimension of 1
        """
        super().__init__()
        if summary_net_dimensions is not None:
            assert critic_net_dimensions is not None
            assert summary_net is None
            assert critic_net is None
            self._summary_net = hk.nets.MLP(
                output_sizes=summary_net_dimensions, activation=jax.nn.relu
            )
            self._critic_net = hk.nets.MLP(
                output_sizes=critic_net_dimensions, activation=jax.nn.relu
            )
        else:
            assert summary_net is not None
            assert critic_net is not None
            self._summary_net = summary_net
            self._critic_net = critic_net

    def __call__(self, method: str, **kwargs):
        """Apply the network.

        Args:
            method: the method to be called
            kwargs: keyword arguments to be passed to the called method
        """
        return getattr(self, "_" + method)(**kwargs)

    def _forward(self, y, theta):
        s = self._summary(y)
        c = self._critic(s, theta)
        return s, c

    def _summary(self, y):
        return self._summary_net(y)

    def _critic(self, y, theta):
        return self._critic_net(jnp.concatenate([y, theta], axis=-1))
