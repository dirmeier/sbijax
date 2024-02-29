from typing import Callable

import distrax
import haiku as hk
import jax
from jax import numpy as jnp
from jax.nn import glu
from scipy import integrate

__all__ = ["CCNF", "make_ccnf"]


class CCNF(hk.Module):
    """Conditional continuous normalizing flow.

    Args:
        n_dimension: the dimensionality of the modelled space
        transform: a haiku module. The transform is a callable that has to
            take as input arguments named 'theta', 'time', 'context' and
            **kwargs. Theta, time and context are two-dimensional arrays
            with the same batch dimensions.
    """

    def __init__(self, n_dimension: int, transform: Callable):
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
        self._network = transform
        self._base_distribution = distrax.Normal(jnp.zeros(n_dimension), 1.0)

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

        def ode_func(time, theta_t):
            theta_t = theta_t.reshape(-1, self._n_dimension)
            time = jnp.full((theta_t.shape[0], 1), time)
            ret = self.vector_field(
                theta=theta_t, time=time, context=context, **kwargs
            )
            return ret.reshape(-1)

        res = integrate.solve_ivp(
            ode_func,
            (0.0, 1.0),
            theta_0.reshape(-1),
            rtol=1e-5,
            atol=1e-5,
            method="RK45",
        )

        ret = res.y[:, -1].reshape(-1, self._n_dimension)
        return ret

    def vector_field(self, theta, time, context, **kwargs):
        """Compute the vector field.

        Args:
            theta: array of parameters
            time: time variables
            context: array of conditioning variables

        Keyword Args:
            keyword arguments that aer passed tothe neural network
        """
        time = jnp.full((theta.shape[0], 1), time)
        return self._network(theta=theta, time=time, context=context, **kwargs)


# pylint: disable=too-many-arguments
class _ResnetBlock(hk.Module):
    """A block for a 1d residual network."""

    def __init__(
        self,
        hidden_size: int,
        activation: Callable = jax.nn.relu,
        dropout_rate: float = 0.2,
        do_batch_norm: bool = False,
        batch_norm_decay: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = activation
        self.do_batch_norm = do_batch_norm
        self.dropout_rate = dropout_rate
        self.batch_norm_decay = batch_norm_decay

    def __call__(self, inputs, context, is_training=False):
        outputs = inputs
        if self.do_batch_norm:
            outputs = hk.BatchNorm(True, True, self.batch_norm_decay)(
                outputs, is_training=is_training
            )
        outputs = hk.Linear(self.hidden_size)(outputs)
        outputs = self.activation(outputs)
        if is_training:
            outputs = hk.dropout(
                rng=hk.next_rng_key(), rate=self.dropout_rate, x=outputs
            )
        outputs = hk.Linear(self.hidden_size)(outputs)
        context_proj = hk.Linear(inputs.shape[-1])(context)
        outputs = glu(jnp.concatenate([outputs, context_proj], axis=-1))
        return outputs + inputs


# pylint: disable=too-many-arguments
class _CCNFResnet(hk.Module):
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

    def __call__(self, theta, time, context, is_training=False, **kwargs):
        outputs = context
        # this is a bit weird, but what the paper suggests:
        # instead of using times and context (i.e., y) as conditioning variables
        # it suggests using times and theta and use y in the resnet blocks,
        # since theta is typically low-dim and y is typically high-dime
        t_theta_embedding = jnp.concatenate(
            [
                hk.Linear(self.n_dimension)(theta),
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


def make_ccnf(
    n_dimension: int,
    n_layers: int = 2,
    hidden_size: int = 64,
    activation: Callable = jax.nn.tanh,
    dropout_rate: float = 0.2,
    do_batch_norm: bool = False,
    batch_norm_decay: float = 0.2,
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
    Returns:
        returns a conditional continuous normalizing flow
    """

    @hk.transform
    def _flow(method, **kwargs):
        nn = _CCNFResnet(
            n_layers=n_layers,
            n_dimension=n_dimension,
            hidden_size=hidden_size,
            activation=activation,
            do_batch_norm=do_batch_norm,
            dropout_rate=dropout_rate,
            batch_norm_decay=batch_norm_decay,
        )
        ccnf = CCNF(n_dimension, nn)
        return ccnf(method, **kwargs)

    return _flow
