from sbijax._src.nle import NLE


# ruff: noqa: PLR0913, E501
class SNLE(NLE):
    """Surjective neural likelihood estimation.

    Implements the method introduced in :cite:t:`dirmeier2023simulation`.
    SNLE is particularly useful when dealing with high-dimensional data since
    it reduces its dimensionality using dimensionality reduction.

    Args:
        model_fns: a tuple of calalbles. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        density_estimator: a (neural) conditional density estimator
            to model the likelihood function

    Examples:
        >>> from jax import numpy as jnp
        >>> from sbijax import SNLE
        >>> from sbijax.nn import make_maf
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...    dict(theta=tfd.Normal(jnp.zeros(5), 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(
        ...     theta["theta"], 1.0).sample(seed=seed, sample_shape=(2,)
        ... ).reshape(-1, 10)
        >>> fns = prior, s
        >>> neural_network = make_maf(10, n_layer_dimensions=[10, 10, 5, 5, 5])
        >>> model = SNLE(fns, neural_network)

    References:
        Dirmeier, Simon, et al. "Simulation-based inference using surjective sequential neural likelihood estimation." arXiv preprint arXiv:2308.01054, 2023.
    """
