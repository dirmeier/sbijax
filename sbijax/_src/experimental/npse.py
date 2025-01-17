from functools import partial

from jax import numpy as jnp
from jax import random as jr

from sbijax import FMPE


def beta_fn(t, beta_max, beta_min):
    return beta_min + t * (beta_max - beta_min)


def integral(t, beta_max, beta_min):
    return beta_min * t + 0.5 * (beta_max - beta_min) * t**2


def _sde(x, t, beta_max, beta_min):
    beta_t = beta_fn(t, beta_max, beta_min)
    intr = integral(t, beta_max, beta_min)
    drift = -0.5 * x * beta_t
    diffusion = 1.0 - jnp.exp(-2.0 * intr)
    diffusion = jnp.sqrt(beta_t * diffusion)
    return drift, diffusion


def _margprob_params(x, t, beta_max, beta_min):
    intr = integral(t, beta_max, beta_min)
    mean = x * jnp.exp(-0.5 * intr)[:, None]
    std = 1.0 - jnp.exp(-intr)
    return mean, std


# ruff: noqa: PLR0913, E501
class NPSE(FMPE):
    """Neural posterior score estimation.

    Implements (truncated sequential) neural posterior score estimation as introduced in
    :cite:t:`sharrock2024sequential`.

    Args:
        model_fns: a tuple of callables. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        density_estimator: a score estimator

    Examples:
        >>> from sbijax.experimental import NPSE
        >>> from sbijax.experimental.nn import make_scorenet
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...    dict(theta=tfd.Normal(0.0, 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> neural_network = make_scorenet(1)
        >>> model = NPSE(fns, neural_network)

    References:
        Sharrock, Louis, et al. "Sequential neural score estimation: likelihood-free inference with conditional score based diffusion models." International Conference on Machine Learning, 2025.
    """

    def __init__(
        self, model_fns, density_estimator, beta_min=0.1, beta_max=10.0
    ):
        super().__init__(model_fns, density_estimator, beta_min)
        self.beta_min = super().sigma_min
        self.beta_max = beta_max

    def get_loss_fn(self):
        p_marg_prob_fn = partial(
            _margprob_params, beta_min=self.beta_min, beta_max=self.beta_max
        )
        sde_fn = partial(_sde, beta_min=self.beta_min, beta_max=self.beta_max)

        def fn(params, rng_key, apply_fn, is_training=True, **batch):
            theta = batch["theta"]
            n, _ = theta.shape

            t_key, rng_key = jr.split(rng_key)
            times = jr.uniform(t_key, shape=(n, 1))

            theta_key, rng_key = jr.split(rng_key)
            theta_t = _sample_theta_t(theta_key, times, theta, sigma_min)

            train_rng, rng_key = jr.split(rng_key)
            vs = apply_fn(
                params,
                train_rng,
                method="vector_field",
                theta=theta_t,
                time=times,
                context=batch["y"],
                is_training=is_training,
            )
            uts = _ut(theta_t, theta, times, sigma_min)

            loss = jnp.mean(jnp.square(vs - uts))
            return loss

        return fn

    def _init_params(self, rng_key, **init_data):
        times = jr.uniform(jr.PRNGKey(0), shape=(init_data["y"].shape[0], 1))
        params = self.model.init(
            rng_key,
            method="vector_field",
            theta=init_data["theta"],
            time=times,
            context=init_data["y"],
            is_training=False,
        )
        return params
