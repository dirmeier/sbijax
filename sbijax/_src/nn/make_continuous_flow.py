from typing import Callable

import distrax
import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from scipy import integrate

__all__ = ["CNF", "make_cnf"]

from sbijax._src.nn.make_resnet import _ResnetBlock


def to_output_shape(x, t):
    new_shape = (-1,) + tuple(np.ones(x.ndim - 1, dtype=np.int32).tolist())
    t = t.reshape(new_shape)
    return t


def sample_theta_t(rng_key, x, times, sigma_min):
    times =  to_output_shape(x, times)
    mus = times * x
    sigmata = 1.0 - (1.0 - sigma_min) * times

    noise = jr.normal(rng_key, shape=x.shape)
    theta_t = noise * sigmata + mus
    return theta_t


def ut(x_t, x, times, sigma_min):
    times = to_output_shape(x, times)
    num = x - (1.0 - sigma_min) * x_t
    denom = 1.0 - (1.0 - sigma_min) * times
    return num / denom


# ruff: noqa: PLR0913,D417
class CNF(hk.Module):
    """Conditional continuous normalizing flow.

    Args:
        n_dimension: the dimensionality of the modelled space
        transform: a haiku module. The transform is a callable that has to
            take as input arguments named 'theta', 'time', 'context' and
            **kwargs. Theta, time and context are two-dimensional arrays
            with the same batch dimensions.
    """

    def __init__(self, n_dimension: int, transform: Callable, sigma_min=0.001):
        """Conditional continuous normalizing flow.

        Args:
            n_dimension: the dimensionality of the modelled space
            transform: a haiku module. The transform is a callable that has to
                take as input arguments named 'theta', 'time', 'context' and
                **kwargs. Theta, time and context are two-dimensional arrays
                with the same batch dimensions.
        """
        super().__init__()
        self._n_dimension = n_dimension
        self._score_model = transform
        self._base_distribution = distrax.Normal(jnp.zeros(n_dimension), 1.0)
        self.sigma_min = sigma_min

    def __call__(self, method, **kwargs):
        """Aplpy the flow.

        Args:
            method (str): method to call

        Keyword Args:
            keyword arguments for the called method:
        """
        return getattr(self, method)(**kwargs)

    def sample(self, context, **kwargs):
        """Sample from the pushforward.

        Args:
            context: array of conditioning variables
        """
        theta_0 = self._base_distribution.sample(
            seed=hk.next_rng_key(), sample_shape=(context.shape[0],)
        )

        def ode_fn(time, theta_t):
            theta_t = theta_t.reshape(-1, self._n_dimension)
            time = jnp.repeat(time, theta_t.shape[0])
            ret = self._score_model(
                inputs=theta_t, time=time, context=context, **kwargs
            )
            return ret.reshape(-1)

        res = integrate.solve_ivp(
            ode_fn,
            (0.0, 1.0),
            theta_0.reshape(-1),
            rtol=1e-5,
            atol=1e-5,
            method="RK45",
        )

        ret = res.y[:, -1].reshape(-1, self._n_dimension)
        return ret

    def loss(self, inputs, context, is_training, **kwargs):
        n, _ = inputs.shape
        times = jr.uniform(hk.next_rng_key(), shape=(n,))
        theta_t = sample_theta_t(
            hk.next_rng_key(), inputs, times, self.sigma_min
        )
        vs = self._score_model(
            inputs=theta_t,
            time=times,
            context=context,
            is_training=is_training,
        )
        uts = ut(theta_t, inputs, times, self.sigma_min)
        loss = jnp.mean(jnp.square(vs - uts))
        return loss


# pylint: disable=too-many-arguments
class _CNFResnet(hk.Module):
    """A simplified 1-d residual network."""

    def __init__(
        self,
        n_layers: int,
        n_dimension: int,
        hidden_size: int,
        activation: Callable = jax.nn.relu,
        dropout_rate: float = 0.2,
        do_batch_norm: bool = True,
        batch_norm_decay: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_dimension = n_dimension
        self.hidden_size = hidden_size
        self.activation = activation
        self.do_batch_norm = do_batch_norm
        self.dropout_rate = dropout_rate
        self.batch_norm_decay = batch_norm_decay

    def __call__(self, inputs, time, context, is_training=False, **kwargs):
        outputs = context
        # this is a bit weird, but what the paper suggests:
        # instead of using times and context (i.e., y) as conditioning variables
        # it suggests using times and theta and use y in the resnet blocks,
        # since theta is typically low-dim and y is typically high-dime
        time = to_output_shape(inputs, time)
        t_theta_embedding = jnp.concatenate(
            [
                hk.Linear(self.n_dimension)(inputs),
                hk.Linear(self.n_dimension)(time),
            ],
            axis=-1,
        )
        outputs = hk.Linear(self.hidden_size)(outputs)
        outputs = self.activation(outputs)
        for _ in range(self.n_layers):
            outputs = _ResnetBlock(
                hidden_size=self.hidden_size,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                do_batch_norm=self.do_batch_norm,
                batch_norm_decay=self.batch_norm_decay,
            )(outputs, context=t_theta_embedding, is_training=is_training)
        outputs = self.activation(outputs)
        outputs = hk.Linear(self.n_dimension)(outputs)
        return outputs


# ruff: noqa: PLR0913
def make_cnf(
    n_dimension: int,
    n_layers: int = 2,
    hidden_size: int = 64,
    activation: Callable = jax.nn.relu,
    dropout_rate: float = 0.1,
    do_batch_norm: bool = False,
    batch_norm_decay: float = 0.2,
    sigma_min: float = 0.001,
):
    """Create a conditional continuous normalizing flow.

    The CCNF uses a residual network as transformer which is created
    automatically.

    Args:
        n_dimension: dimensionality of modelled space
        n_layers: number of resnet blocks
        hidden_size: sizes of hidden layers for each resnet block
        activation: a jax activation function
        dropout_rate: dropout rate to use in resnet blocks
        do_batch_norm: use batch normalization or not
        batch_norm_decay: decay rate of EMA in batch norm layer
        sigma_min: minimal scaling for the vector field
    Returns:
        returns a conditional continuous normalizing flow
    """

    @hk.transform
    def _flow(method, **kwargs):
        nn = _CNFResnet(
            n_layers=n_layers,
            n_dimension=n_dimension,
            hidden_size=hidden_size,
            activation=activation,
            do_batch_norm=do_batch_norm,
            dropout_rate=dropout_rate,
            batch_norm_decay=batch_norm_decay,
        )
        cnf = CNF(n_dimension, nn, sigma_min)
        return cnf(method, **kwargs)

    return _flow
