from functools import partial

from jax import numpy as jnp
from jax import random as jr

from sbijax import FMPE




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
        sde: can be either of 'vp' and 've'. Defines the type of SDE to be used as a forward process.
            See the original publication and references therein for details.
        beta_min: beta min. Again, see the paper please.
        beta_max: beta max. Again, see the paper please.
        time_eps: some small number to use as minimum time point for the forward process. Used for numerical
            stability.
        time_max: maximum integration time. 1 is good, but so is 5 or 10.

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
        self, model_fns, density_estimator, sde="vp", beta_min=0.1, beta_max=10.0, time_eps=0.001, time_max=1
    ):
        super().__init__(model_fns, density_estimator, beta_min)
        self.sde = sde
        self.beta_min = super().sigma_min
        self.beta_max = beta_max
        self.time_eps = time_eps
        self.time_max = time_max

    def get_loss_fn(self):
        marg_prob_params = get_margprob_params_fn(
            self.sde, beta_min=self.beta_min, beta_max=self.beta_max
        )

        def fn(params, rng_key, apply_fn, is_training, **batch):
            theta, y = batch["theta"], batch["y"]

            time_key, noise_key, train_rng = jr.split(rng_key, 3)
            time = jr.uniform(time_key, (theta.shape[0],), minval=self.time_eps, maxval=self.time_max)

            noise = jr.normal(noise_key, theta.shape)
            mean, scale = marg_prob_params(theta, time)
            theta_t = mean + noise * scale[:, None]
            score = apply_fn(
                params,
                train_rng,
                method="vector_field",
                theta=theta_t,
                time=time,
                context=y,
                is_training=is_training
            )

            loss = jnp.sum((score * scale[:, None] + noise) ** 2, axis=-1)
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
