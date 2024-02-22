from typing import Callable, List

import haiku as hk
import jax
from jax import numpy as jnp

from sbijax._src.nn.snass_net import SNASSNet


# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-arguments
class SNASSSNet(SNASSNet):
    """A network for SNASSS."""

    def __init__(
        self,
        summary_net_dimensions: List[int] = None,
        sec_summary_net_dimensions: List[int] = None,
        critic_net_dimensions: List[int] = None,
        summary_net: Callable = None,
        sec_summary_net: Callable = None,
        critic_net: Callable = None,
    ):
        """Constructs a SNASSSNet.

        Can be used either by providing network dimensions or haiku modules.

        Args:
            summary_net_dimensions: a list of integers representing
                the dimensionalities of the summary network. The _last_
                dimension determines the dimensionality of the summary statistic
            sec_summary_net_dimensions: a list of integers representing
                the dimensionalities of the second summary network. The _last_
                should be 1.
            critic_net_dimensions: a list of integers representing the
                dimensionality of the critic network. The _last_ dimension
                needs to be 1.
            summary_net: a haiku MLP with trailing dimension being the
                dimensionality of the summary statistic
            sec_summary_net: a haiku MLP with trailing dimension of 1
            critic_net: : a haiku MLP with a trailing dimension of 1
        """
        super().__init__(
            summary_net_dimensions,
            critic_net_dimensions,
            summary_net,
            critic_net,
        )
        if sec_summary_net_dimensions is not None:
            assert sec_summary_net is None
            self._sec_summary_net = hk.nets.MLP(
                output_sizes=sec_summary_net_dimensions, activation=jax.nn.relu
            )
        else:
            self._sec_summary_net = sec_summary_net

    def __call__(self, method: str, **kwargs):
        """Apply the network.

        Args:
          method: the method to be called
          kwargs: keyword arguments to be passed to the called method
        """
        return getattr(self, "_" + method)(**kwargs)

    def _forward(self, y, theta):
        s = self._summary(y)
        s2 = self._secondary_summary(s, theta)
        c = self._critic(s2, y[:, [0]])
        return s, s2, c

    def _secondary_summary(self, y, theta):
        return self._sec_summary_net(jnp.concatenate([y, theta], axis=-1))
