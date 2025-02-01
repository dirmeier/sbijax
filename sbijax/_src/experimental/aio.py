import jax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax import FMPE, as_inference_data, inference_data_as_dictionary


# ruff: noqa: PLR0913, E501
class AiO(FMPE):
    """All-in-one simulation-based inference.

    Implements all-on-one posterior estimation as introduced
    :cite:t:`gloeckler2024allinone`.

    Args:
        model_fns: a tuple of callables. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        density_estimator: a score estimator

    Examples:
        >>> from sbijax.experimental import AiO
        >>> from sbijax.experimental.nn import make_simformer_based_score_model
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...    dict(theta=tfd.Normal(jnp.zeros(2), 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> neural_network = make_simformer_based_score_model(2, jnp.eye(4))
        >>> model = AiO(fns, neural_network)

    References:
        Gloeckler, Manuel, et al. "All-in-one simulation-based inference." International Conference on Machine Learning, 2024.
    """

    def _simulate_parameters_with_model(
        self, rng_key, params, observable, *, n_samples=4_000, **kwargs
    ):
        prior_key, rng_key = jr.split(rng_key)
        prior_fn = self.get_truncated_prior(
            prior_key, params, observable, n_samples=int(1e5)
        )
        return prior_fn(rng_key, n_samples)

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
        samp = self.prior_sampler_fn(seed=jr.PRNGKey(0), sample_shape=())
        _, unravel_fn = ravel_pytree(samp)

        sample_key, rng_key = jr.split(rng_key)
        inf_data, _ = self.sample_posterior(
            sample_key, params, observable, n_samples=n_samples
        )
        posterior_samples = inference_data_as_dictionary(inf_data.posterior)
        lp_key, rng_key = jr.split(rng_key)
        flat_posterior_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(
            posterior_samples
        )
        log_probs = self.model.apply(
            params,
            rng=lp_key,
            method="log_prob",
            inputs=flat_posterior_samples,
            context=jnp.tile(observable, [flat_posterior_samples.shape[0], 1]),
            is_training=False,
        )
        trunc_boundary = jnp.quantile(log_probs, 5e-4)
        min_posterior, max_posterior = (
            jax.tree.map(lambda x: x.min(axis=0), posterior_samples),
            jax.tree.map(lambda x: x.max(axis=0), posterior_samples),
        )
        sample_key, rng_key = jr.split(rng_key)
        prior_samples = self.prior_sampler_fn(
            seed=sample_key, sample_shape=(int(1e6),)
        )
        min_prior, max_prior = (
            jax.tree.map(lambda x: x.min(axis=0), prior_samples),
            jax.tree.map(lambda x: x.max(axis=0), prior_samples),
        )
        hypercube_min = jax.tree.map(
            lambda po, pr: jnp.concatenate(
                [po[None, ...], pr[None, ...]], axis=0
            ).max(axis=0),
            min_posterior,
            min_prior,
        )
        hypercube_max = jax.tree.map(
            lambda po, pr: jnp.concatenate(
                [po[None, ...], pr[None, ...]], axis=0
            ).min(axis=0),
            max_posterior,
            max_prior,
        )

        def hypercube_uniform_prior(rng_key, n_samples):
            return jr.uniform(
                rng_key,
                (
                    n_samples,
                    flat_posterior_samples.shape[-1],
                ),
                minval=jnp.concatenate(jax.tree.leaves(hypercube_min)),
                maxval=jnp.concatenate(jax.tree.leaves(hypercube_max)),
            )

        def truncated_prior_fn(rng_key, n_samples, n_iter=1_000):
            cnt = n_curr = 0
            samples_out = []
            while n_curr < n_samples and cnt < n_iter:
                sample_key, lp_key, rng_key = jr.split(rng_key, 3)
                samples = hypercube_uniform_prior(sample_key, n_samples)
                log_probs = self.model.apply(
                    params,
                    rng=lp_key,
                    method="log_prob",
                    inputs=samples,
                    context=jnp.tile(observable, [samples.shape[0], 1]),
                    is_training=False,
                )
                accepted_samples = samples[log_probs > trunc_boundary]
                samples_out.append(accepted_samples)
                n_curr += len(accepted_samples)
                cnt += 1

            if cnt == n_iter:
                raise ValueError("truncated sampling did not converge")
            thetas = jnp.concatenate(samples_out, axis=0)[:n_samples]

            def reshape(p):
                if p.ndim == 1:
                    p = p.reshape(p.shape[0], 1)
                p = p.reshape(1, *p.shape)
                return p

            ess = n_curr / (cnt * n_samples)
            thetas = jax.tree_map(
                reshape, jax.vmap(unravel_fn)(thetas[:n_samples])
            )
            inference_data = as_inference_data(thetas, jnp.squeeze(observable))

            return inference_data, ess

        return truncated_prior_fn
