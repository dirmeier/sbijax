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

    Examples:
        >>> from sbijax.experimental import NPSE
        >>> from sbijax.experimental.nn import make_score_model
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...    dict(theta=tfd.Normal(0.0, 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> neural_network = make_score_model(1)
        >>> model = NPSE(fns, neural_network)

    References:
        Sharrock, Louis, et al. "Sequential neural score estimation: likelihood-free inference with conditional score based diffusion models." International Conference on Machine Learning, 2025.
    """

    def __init__(self, model_fns, density_estimator):
        super().__init__(model_fns, density_estimator)

    def _init_params(self, rng_key, **init_data):
        params = self.model.init(
            rng_key,
            method="loss",
            inputs=init_data["theta"],
            context=init_data["y"],
            is_training=False,
        )
        return params

    def get_truncated_prior(self, rng_key, params, observable, n_samples):
        sample_key, rng_key = jr.split(rng_key)
        posterior_samples = self.sample_posterior(
            sample_key, params, observable, n_samples=n_samples
        )
        min_posterior, max_posterior = (
            posterior_samples.min(axis=0),
            posterior_samples.max(axis=0),
        )
        log_probs = self.model.log_prob(posterior_samples, observable)
        trunc_boundary = jnp.quantile(log_probs, 5e-4)

        sample_key, rng_key = jr.split(rng_key)
        prior_samples = self.prior_sampler_fn(
            seed=sample_key, sample_shape=(1e6,)
        )
        min_prior, max_prior = (
            prior_samples.min(axis=0),
            prior_samples.max(axis=0),
        )

        hypercube_min = jnp.concatenate(
            [min_posterior[None, :], min_prior[None, :]], axis=0).max(axis=0)
        hypercube_max = jnp.concatenate(
            [max_posterior[None, :], max_prior[None, :]], axis=0
        ).min(axis=0)

        def hypercube_uniform_prior(rng_key, n_samples):
            return jr.uniform(
                rng_key,
                (n_samples, len(min_prior)),
                minval=hypercube_min,
                maxval=hypercube_max,
            )

        def truncated_prior_fn(rng_key, n_samples, n_iter=1_000):
            cnt = n_curr = 0
            samples_out = []
            while n_curr < n_samples and cnt < n_iter:
                sample_key, rng_key = jr.split(rng_key, 3)
                samples = hypercube_uniform_prior(sample_key, n_samples)
                log_probs = self.model.log_prob(samples, observable)
                accepted_samples = samples[log_probs > trunc_boundary]
                samples_out.append(accepted_samples)
                n_curr += len(accepted_samples)
                cnt += 1

            if cnt == n_iter:
                assert ValueError("truncated sampling did not converve")
            return jnp.concatenate(samples_out)[:n_samples]

        return truncated_prior_fn
