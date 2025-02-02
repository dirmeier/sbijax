from functools import partial
from typing import Callable

import chex
import diffrax as dfx
import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from scipy import integrate
from tensorflow_probability.substrates.jax import distributions as tfd

__all__ = ["make_score_model", "ScoreModel", "timestep_embedding"]


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
    def ve(inputs, time):
        time = to_output_shape(inputs, time)
        sigma = beta_min * (beta_max / beta_min) ** time
        drift = jnp.zeros_like(inputs)
        diffusion = sigma * jnp.sqrt(
            2 * (jnp.log(beta_max) - jnp.log(beta_min))
        )
        return drift, diffusion

    def vp(inputs, time):
        time = to_output_shape(inputs, time)
        beta_t = beta_min + time * (beta_max - beta_min)
        drift = -0.5 * beta_t * inputs
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
    def ve(inputs, time):
        time = to_output_shape(inputs, time)
        mean = inputs
        std = beta_min * (beta_max / beta_min) ** time
        return mean, std

    def vp(inputs, time):
        time = to_output_shape(inputs, time)
        beta = -0.25 * time**2 * (beta_max - beta_min) - 0.5 * time * beta_min
        mean = jnp.exp(beta) * inputs
        std = jnp.sqrt(1.0 - jnp.exp(2.0 * beta))
        return mean, std

    match sde:
        case "ve":
            return ve
        case "vp":
            return vp
        case _:
            raise ValueError("incorrect sde given: choose from ['ve', 'vp']")


def get_log_prob_fn(apply_fn, is_training, sde, beta_max, beta_min):
    def ve(inputs, time, context):
        drift_and_diffusion_fn = get_forward_drift_and_diffusion_fn(
            sde, beta_max, beta_min
        )

        def drift_fn(inputs_t):
            drift, diffusion = drift_and_diffusion_fn(inputs, time)
            score = apply_fn(
                inputs=inputs_t,
                time=time,
                context=context,
                is_training=is_training,
            )
            ret = drift - 0.5 * diffusion**2 * score
            return ret

        drift, vjp_fn = jax.vjp(drift_fn, inputs)
        (dfdtheta,) = jax.vmap(vjp_fn)(jnp.eye(inputs.shape[0]))
        dlogp = jnp.trace(dfdtheta)
        return drift, dlogp

    def vp(inputs, time, context):
        drift_and_diffusion_fn = get_forward_drift_and_diffusion_fn(
            sde, beta_max, beta_min
        )

        def drift_fn(inputs_t):
            drift, diffusion = drift_and_diffusion_fn(inputs, time)
            score = apply_fn(
                inputs=inputs_t,
                time=time,
                context=context,
                is_training=is_training,
            )
            ret = drift - 0.5 * diffusion**2 * score
            return ret

        drift, vjp_fn = jax.vjp(drift_fn, inputs)
        (dfdtheta,) = jax.vmap(vjp_fn)(jnp.eye(inputs.shape[0]))
        dlogp = jnp.trace(dfdtheta)
        return drift, dlogp

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
        time_delta=0.01,
    ):
        super().__init__()
        self._n_dimension = n_dimension
        self._score_net = transform
        self._sde = sde
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._time_eps = time_eps
        self._time_max = time_max
        self._time_delta = time_delta
        self._base_distribution = tfd.Independent(
            tfd.Normal(jnp.zeros(n_dimension), 1.0), 1
        )

    def __call__(self, method, **kwargs):
        """Apply the model.

        Args:
            method (str): method to call

        Keyword Args:
            keyword arguments for the called method:
        """
        return getattr(self, method)(**kwargs)

    def sample(self, context, **kwargs):
        """Sample from the score model.

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
        """Loss function..

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

    def log_prob(self, inputs, context, is_training, **kwargs):
        """Log-probability of an input.

        Args:
            inputs: array of inputs
            context: array of conditioning variables

        Keyword Args:
            keyword arguments that aer passed to the neural network
        """
        fn = partial(self._log_prob, is_training=is_training)
        ret = jax.vmap(fn)(inputs, context)
        return ret

    def _log_prob(self, inputs, context, is_training):
        drift_and_diffusion_fn = get_forward_drift_and_diffusion_fn(
            self._sde, self._beta_max, self._beta_min
        )

        def ode_lp_fn(time, inputs_t, args):
            inputs_t, _ = inputs_t
            time = jnp.atleast_1d(time)

            def drift_fn(inputs):
                drift, diffusion = drift_and_diffusion_fn(
                    jnp.atleast_2d(inputs), time
                )
                score = self._score_net(
                    inputs=jnp.atleast_2d(inputs),
                    time=time,
                    context=jnp.atleast_2d(context),
                    is_training=is_training,
                )
                ret = drift - 0.5 * diffusion**2 * score
                return ret.squeeze()

            drift, vjp_fn = jax.vjp(drift_fn, inputs_t)
            (dfdtheta,) = jax.vmap(vjp_fn)(jnp.eye(inputs_t.shape[0]))
            dlogp = jnp.trace(dfdtheta)
            return drift, dlogp

        term = dfx.ODETerm(ode_lp_fn)
        solver = dfx.Tsit5()
        sol = dfx.diffeqsolve(
            term,
            solver,
            self._time_eps,
            self._time_max,
            self._time_delta,
            (inputs, 0.0),
        )
        (latents,), (delta_log_likelihood,) = sol.ys
        lp = self._base_distribution.log_prob(latents)
        return delta_log_likelihood + lp


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
    """Create a score model for NPSE.

    The score model uses MLPs to embed the data, the parameters and the time
    points (after projecting them with a sinusoidal embedding).
    The score net itself is also an MLP.

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
        returns a score model that can be used for inference using NPSE.
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
