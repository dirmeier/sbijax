from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


# ruff: noqa: PLR0913, E501
def tree():
    """Tree model.

    Constructs prior, simulator and likelihood functions.

    Returns:
      returns a tuple of three objects. The first is a
      tfd.JointDistributionNamed serving as a prior distribution. The second
      is a simulator function that can be used to generate data. The third
      is None (since the likelihood is intractable and to be consistent with other
      models).

    References:
      Gloeckler, Manuel, et al., All-in-one simulation-based inference, 2025
    """

    def a_prior_fn():
        return tfd.Independent(
            tfd.Normal(jnp.zeros(1), 1.0),
            reinterpreted_batch_ndims=1,
        )

    def b_prior_fn(**kwargs):
        return tfd.Independent(tfd.Normal(kwargs["a"], 1.0), 1)

    def c_prior_fn(**kwargs):
        return tfd.Independent(tfd.Normal(kwargs["a"], 1.0), 1)

    def prior_fn():
        prior = tfd.JointDistributionNamed(
            dict(a=a_prior_fn, b=b_prior_fn, c=c_prior_fn)
        )
        return prior

    def likelihood(y, theta):
        a, b, c = theta["a"], theta["b"], theta["c"]
        lik_fn = tfd.Independent(
            tfd.Normal(
                jnp.concatenate(
                    [jnp.sin(b) ** 2, 0.1 * b**2, 0.1 * c**2, jnp.cos(b) ** 2],
                    axis=-1,
                ),
                jnp.array([0.2, 0.2, 0.6, 0.1]),
            ),
            reinterpreted_batch_ndims=1,
        )
        log_lik = lik_fn.log_prob(y)
        return log_lik

    def simulator(seed, theta):
        a, b, c = theta["a"], theta["b"], theta["c"]
        sim_fn = tfd.Independent(
            tfd.Normal(
                jnp.concatenate(
                    [jnp.sin(b) ** 2, 0.1 * b**2, 0.1 * c**2, jnp.cos(b) ** 2],
                    axis=-1,
                ),
                jnp.array([0.2, 0.2, 0.6, 0.1]),
            ),
            reinterpreted_batch_ndims=1,
        )
        return sim_fn.sample(seed=seed).reshape(-1, 4)

    return prior_fn(), simulator, likelihood
