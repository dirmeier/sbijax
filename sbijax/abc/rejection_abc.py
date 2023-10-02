from jax import numpy as jnp
from jax import random as jr

from sbijax._sbi_base import SBI


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-few-public-methods
class RejectionABC(SBI):
    """
    Sisson et al. - Handbook of approximate Bayesian computation

    Algorithm 4.1, "ABC Rejection Sampling Algorithm"
    """

    def __init__(self, model_fns, summary_fn, kernel_fn):
        super().__init__(model_fns)
        self.kernel_fn = kernel_fn
        self.summary_fn = summary_fn

    # pylint: disable=arguments-differ
    def sample_posterior(
        self,
        rng_key,
        observable,
        n_samples,
        n_simulations_per_theta,
        K,
        h,
        **kwargs,
    ):
        """
        Sample from the approximate posterior

        Parameters
        ----------
        rng_key: jax.PRNGKey
            a random key
        observable: jnp.Array
            observation to condition on
        n_samples: int
            number of samples to draw for each parameter
        n_simulations_per_theta: int
            number of simulations for each paramter sample
        K: double
            normalisation parameter
        h: double
            kernel scale

        Returns
        -------
        chex.Array
            an array of samples from the posterior distribution of dimension
            (n_samples \times p)
        """

        observable = jnp.atleast_2d(observable)

        thetas = None
        n = n_samples
        K = jnp.maximum(
            K, self.kernel_fn(jnp.zeros((1, 2, 2)), jnp.zeros((1, 2, 2)))[0]
        )
        while n > 0:
            p_key, simulate_key, prior_key, rng_key = jr.split(rng_key)
            n_sim = jnp.minimum(n, 1000)
            ps = self.prior_sampler_fn(seed=prior_key, sample_shape=(n_sim,))
            ys = self.simulator_fn(
                seed=simulate_key,
                theta=jnp.tile(ps, [n_simulations_per_theta, 1, 1]),
            )
            ys = jnp.swapaxes(ys, 1, 0)
            k = self.kernel_fn(
                self.summary_fn(ys), self.summary_fn(observable), h
            )
            p = jr.uniform(p_key, shape=(len(k),))
            mr = k / K
            idxs = jnp.where(p < mr)[0]
            if thetas is None:
                thetas = ps[idxs]
            else:
                thetas = jnp.vstack([thetas, ps[idxs]])
            n -= len(idxs)
        return thetas[:n_samples, :]
