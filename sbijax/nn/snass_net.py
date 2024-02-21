from typing import Callable, List

import haiku as hk
import jax
from jax import numpy as jnp


# pylint: disable=missing-function-docstring,missing-class-docstring
class SNASSNet(hk.Module):
    """
    A network for SNASS
    """

    def __init__(
        self,
        summary_net_dimensions: List[int] = None,
        critic_net_dimensions: List[int] = None,
        summary_net: Callable = None,
        critic_net: Callable = None,
    ):
        super().__init__()
        if summary_net_dimensions is not None:
            assert critic_net_dimensions is not None
            assert summary_net is None
            assert critic_net is None
            self._summary = hk.nets.MLP(
                output_sizes=summary_net_dimensions, activation=jax.nn.relu
            )
            self._critic = hk.nets.MLP(
                output_sizes=critic_net_dimensions, activation=jax.nn.relu
            )
        else:
            assert summary_net is not None
            assert critic_net is not None
            self._summary = summary_net
            self._critic = critic_net

    def __call__(self, method, **kwargs):
        return getattr(self, method)(**kwargs)

    def forward(self, y, theta):
        s = self.summary(y)
        c = self.critic(s, theta)
        return s, c

    def summary(self, y):
        return self._summary(y)

    def critic(self, y, theta):
        return self._critic(jnp.concatenate([y, theta], axis=-1))
