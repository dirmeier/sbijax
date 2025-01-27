from typing import Callable

import chex
import distrax
import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from scipy import integrate

__all__ = ["make_score_model", "ScoreModel"]


def to_output_shape(x, t):
    new_shape = (-1,) + tuple(np.ones(x.ndim - 1, dtype=np.int32).tolist())
    t = t.reshape(new_shape)
    return t


def timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
    half = embedding_dim // 2
    freqs = jnp.exp(-jnp.log(10_000) * jnp.arange(0, half) / half)
    emb = timesteps.astype(dtype)[:, None] * freqs[None, ...]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb


def get_forward_drift_and_diffusion_fn(sde, beta_max, beta_min):
    def ve(x, t):
        t = to_output_shape(x, t)
        sigma = beta_min * (beta_max / beta_min) ** t
        drift = jnp.zeros_like(x)
        diffusion = sigma * jnp.sqrt(
            2 * (jnp.log(beta_max) - jnp.log(beta_min))
        )
        return drift, diffusion

    def vp(x, t):
        t = to_output_shape(x, t)
        beta_t = beta_min + t * (beta_max - beta_min)
        drift = -0.5 * beta_t * x
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    match sde:
        case "ve":
            return ve
        case "vp":
            return vp
        case _:
            raise ValueError("incorrect sde given: choose from ['ve', 'vp']")


def get_margprob_params_fn(sde, beta_max, beta_min):
    def ve(x, t):
        t = to_output_shape(x, t)
        mean = x
        std = beta_min * (beta_max / beta_min) ** t
        return mean, std

    def vp(x, t):
        t = to_output_shape(x, t)
        beta = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
        mean = jnp.exp(beta) * x
        std = jnp.sqrt(1.0 - jnp.exp(2.0 * beta))
        return mean, std

    match sde:
        case "ve":
            return ve
        case "vp":
            return vp
        case _:
            raise ValueError("incorrect sde given: choose from ['ve', 'vp']")


# pylint: disable=too-many-arguments
class _ScoreNet(hk.Module):
    """A simplified 1-d residual network."""

    def __init__(
        self,
        n_dimension,
        hidden_sizes,
        data_embedding_layers,
        param_embedding_layers,
        time_embedding_layers,
        activation,
    ):
        super().__init__()
        self.n_dimension = n_dimension
        self.hidden_sizes = hidden_sizes
        self.data_embedding_layers = data_embedding_layers
        self.param_embedding_layers = param_embedding_layers
        self.time_embedding_layers = time_embedding_layers
        self.activation = activation

    def __call__(self, inputs, time, context, **kwargs):
        inputs = hk.nets.MLP(
            self.param_embedding_layers, activation=self.activation
        )(inputs)
        context = hk.nets.MLP(
            self.data_embedding_layers, activation=self.activation
        )(context)
        time = hk.Sequential(
            [
                lambda x: timestep_embedding(x, self.time_embedding_layers[0]),
                hk.nets.MLP(
                    self.time_embedding_layers, activation=self.activation
                ),
            ]
        )(time)

        hidden = jnp.concatenate([inputs, context, time], axis=-1)
        outputs = hk.nets.MLP(
            self.hidden_sizes + (self.n_dimension,), activation=self.activation
        )(hidden)
        return outputs


# ruff: noqa: PLR0913,D417
class ScoreModel(hk.Module):
    """Conventional score model.

    Args:
        n_dimension: the dimensionality of the modelled space
        transform: a haiku module. The transform is a callable that has to
            take as input arguments named 'theta', 'time', 'context' and
            **kwargs. Theta, time and context are two-dimensional arrays
            with the same batch dimensions.
    """

    def __init__(
        self,
        n_dimension: int,
        transform: Callable,
        sde,
        beta_min,
        beta_max,
        time_eps,
        time_max,
    ):
        super().__init__()
        self._n_dimension = n_dimension
        self._score_net = transform
        self._sde = sde
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._time_eps = time_eps
        self._time_max = time_max
        self._base_distribution = distrax.Normal(jnp.zeros(n_dimension), 1.0)

    def __call__(self, method, **kwargs):
        """Apply the model.

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
        drift_and_diffusion_fn = get_forward_drift_and_diffusion_fn(
            self._sde, beta_min=self._beta_min, beta_max=self._beta_max
        )

        def sde_fn(time, theta_t):
            theta_t = theta_t.reshape(-1, self._n_dimension)
            time = jnp.repeat(time, theta_t.shape[0])
            drift, diffusion = drift_and_diffusion_fn(theta_t, time)
            score = self._score_net(
                inputs=theta_t, time=time, context=context, **kwargs
            )
            ret = drift - 0.5 * diffusion**2 * score
            return ret.reshape(-1)

        res = integrate.solve_ivp(
            sde_fn,
            (self._time_max, self._time_eps),
            theta_0.reshape(-1),
            rtol=1e-5,
            atol=1e-5,
            method="RK45",
        )

        ret = res.y[:, -1].reshape(-1, self._n_dimension)
        return ret

    def loss(self, inputs, context, is_training, **kwargs):
        """Compute the vector field (or the score).

        Args:
            inputs: array of inputs
            context: array of conditioning variables

        Keyword Args:
            keyword arguments that aer passed to the neural network
        """
        marg_prob_params = get_margprob_params_fn(
            self._sde, beta_min=self._beta_min, beta_max=self._beta_max
        )
        time = jr.uniform(
            hk.next_rng_key(),
            (inputs.shape[0],),
            minval=self._time_eps,
            maxval=self._time_max,
        )

        noise = jr.normal(hk.next_rng_key(), inputs.shape)
        mean, scale = marg_prob_params(inputs, time)
        theta_t = mean + noise * scale
        score = self._score_net(
            inputs=theta_t, time=time, context=context, is_training=is_training
        )
        chex.assert_equal_shape([noise, score])
        loss = jnp.sum((score * scale + noise) ** 2, axis=-1)
        return loss


# ruff: noqa: PLR0913
def make_score_model(
    n_dimension: int,
    hidden_sizes: tuple[int, ...] = (128, 128),
    data_embedding_layers: tuple[int, ...] = (128, 128),
    param_embedding_layers: tuple[int, ...] = (128, 128),
    time_embedding_layers: tuple[int, ...] = (128, 128),
    activation: Callable = jax.nn.relu,
    sde="vp",
    beta_min=0.1,
    beta_max=10.0,
    time_eps=0.001,
    time_max=1,
):
    """Create a score network for NPSE.

    The score network uses MLPs to embed the data, the parameters and the time
    points (after projecting them with a sinusoidal embedding).
    The score net itself is also a MLP.

    Args:
        n_dimension: dimensionality of modelled space
        hidden_sizes: tuple of ints determining the layers of the score network
        data_embedding_layers: a tuple if ints determining the output sizes of
            the data embedding network
        param_embedding_layers: a tuple if ints determining the output sizes of
            the data embedding network
        time_embedding_layers: a tuple if ints determining the output sizes of
            the data embedding network
        activation: a jax activation function
        sde: can be either of 'vp' and 've'. Defines the type of SDE to be used
            as a forward process. See the original publication and references
            therein for details.
        beta_min: beta min. Again, see the paper please.
        beta_max: beta max. Again, see the paper please.
        time_eps: some small number to use as minimum time point for the
            forward process. Used for numerical stability.
        time_max: maximum integration time. 1 is good, but so is 5 or 10.

    Returns:
        returns a conditional continuous normalizing flow
    """

    @hk.transform
    def _score_model(method, **kwargs):
        nn = _ScoreNet(
            n_dimension=n_dimension,
            hidden_sizes=hidden_sizes,
            data_embedding_layers=data_embedding_layers,
            param_embedding_layers=param_embedding_layers,
            time_embedding_layers=time_embedding_layers,
            activation=activation,
        )
        net = ScoreModel(
            n_dimension, nn, sde, beta_min, beta_max, time_eps, time_max
        )
        return net(method, **kwargs)

    return _score_model
