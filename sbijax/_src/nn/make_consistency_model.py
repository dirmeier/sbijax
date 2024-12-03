from typing import Callable

import haiku as hk
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

__all__ = ["ConsistencyModel", "make_cm"]

from sbijax._src.nn.make_resnet import _ResnetBlock


# ruff: noqa: PLR0913,D417
class ConsistencyModel(hk.Module):
    """A consistency model.

    Args:
        n_dimension: the dimensionality of the modelled space
        transform: a haiku module. The transform is a callable that has to
            take as input arguments named 'theta', 'time', 'context' and
            **kwargs. Theta, time and context are two-dimensional arrays
            with the same batch dimensions.
        t_min: minimal time point for ODE integration
        t_max: maximal time point for ODE integration
    """

    def __init__(
        self,
        n_dimension: int,
        transform: Callable,
        t_min: float = 0.001,
        t_max: float = 50.0,
    ):
        """Construct a consistency model.

        Args:
            n_dimension: the dimensionality of the modelled space
            transform: a haiku module. The transform is a callable that has to
                take as input arguments named 'theta', 'time', 'context' and
                **kwargs. Theta, time and context are two-dimensional arrays
                with the same batch dimensions.
            t_min: minimal time point for ODE integration
            t_max: maximal time point for ODE integration
        """
        super().__init__()
        self._n_dimension = n_dimension
        self._network = transform
        self._t_max = t_max
        self._t_min = t_min
        self._base_distribution = tfd.Normal(jnp.zeros(n_dimension), 1.0)

    def __call__(self, method, **kwargs):
        """Aplpy the flow.

        Args:
            method (str): method to call

        Keyword Args:
            keyword arguments for the called method:
        """
        return getattr(self, method)(**kwargs)

    def sample(self, context, **kwargs):
        """Sample from the consistency model.

        Args:
            context: array of conditioning variables
            kwargs: keyword argumente like 'is_training'
        """
        noise = self._base_distribution.sample(
            seed=hk.next_rng_key(), sample_shape=(context.shape[0],)
        )
        y_hat = self.vector_field(noise, self._t_max, context, **kwargs)

        noise = self._base_distribution.sample(
            seed=hk.next_rng_key(), sample_shape=(y_hat.shape[0],)
        )
        tme = self._t_min + (self._t_max - self._t_min) / 2
        noise = jnp.sqrt(jnp.square(tme) - jnp.square(self._t_min)) * noise
        y_tme = y_hat + noise
        y_hat = self.vector_field(y_tme, tme, context, **kwargs)

        return y_hat

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


# pylint: disable=too-many-arguments,too-many-instance-attributes
class _CMResnet(hk.Module):
    """A simplified 1-d residual network."""

    def __init__(
        self,
        n_layers: int,
        n_dimension: int,
        hidden_size: int,
        activation: Callable = jax.nn.relu,
        dropout_rate: float = 0.0,
        do_batch_norm: bool = False,
        batch_norm_decay: float = 0.1,
        t_min: float = 0.001,
        sigma_data: float = 1.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_dimension = n_dimension
        self.hidden_size = hidden_size
        self.activation = activation
        self.do_batch_norm = do_batch_norm
        self.dropout_rate = dropout_rate
        self.batch_norm_decay = batch_norm_decay
        self.sigma_data = sigma_data
        self.var_data = self.sigma_data**2
        self.t_min = t_min

    def __call__(self, theta, time, context, is_training, **kwargs):
        outputs = context
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

        # TODO(simon): dan we choose sigma automatically?
        out_skip = self._c_skip(time) * theta + self._c_out(time) * outputs
        return out_skip

    def _c_skip(self, time):
        return self.var_data / ((time - self.t_min) ** 2 + self.var_data)

    def _c_out(self, time):
        return (
            self.sigma_data
            * (time - self.t_min)
            / jnp.sqrt(self.var_data + time**2)
        )


# ruff: noqa: PLR0913
def make_cm(
    n_dimension: int,
    n_layers: int = 2,
    hidden_size: int = 64,
    activation: Callable = jax.nn.tanh,
    dropout_rate: float = 0.2,
    do_batch_norm: bool = False,
    batch_norm_decay: float = 0.2,
    t_min: float = 0.001,
    t_max: float = 50.0,
    sigma_data: float = 1.0,
):
    """Create a consistency model.

    The consistency model uses a residual network as score network.

    Args:
        n_dimension: dimensionality of modelled space
        n_layers: number of resnet blocks
        hidden_size: sizes of hidden layers for each resnet block
        activation: a jax activation function
        dropout_rate: dropout rate to use in resnet blocks
        do_batch_norm: use batch normalization or not
        batch_norm_decay: decay rate of EMA in batch norm layer
        t_min: minimal time point for ODE integration
        t_max: maximal time point for ODE integration
        sigma_data: the standard deviation of the data :)

    Returns:
        a consistency model
    """

    @hk.transform
    def _cm(method, **kwargs):
        nn = _CMResnet(
            n_layers=n_layers,
            n_dimension=n_dimension,
            hidden_size=hidden_size,
            activation=activation,
            do_batch_norm=do_batch_norm,
            dropout_rate=dropout_rate,
            batch_norm_decay=batch_norm_decay,
            t_min=t_min,
            sigma_data=sigma_data,
        )
        cm = ConsistencyModel(n_dimension, nn, t_min=t_min, t_max=t_max)
        return cm(method, **kwargs)

    return _cm
