import chex
from jax import numpy as jnp
from jax import random

from sbijax._sbi_base import SBI


# pylint: disable=too-many-instance-attributes
class RejectionABC(SBI):
    """
    Sisson et al. - Handbook of approximate Bayesian computation

    Algorithm 4.1, "ABC Rejection Sampling Algorithm"
    """

    def __init__(self, model_fns, summary_fn, kernel_fn):
        super().__init__(model_fns)
        self.kernel_fn = kernel_fn
        self.summary_fn = summary_fn
        self.summarized_observed: chex.Array

    def fit(self, rng_key, observed, **kwargs):
        super().fit(rng_key, observed)
        self.summarized_observed = self.summary_fn(self.observed)

    # pylint: disable=arguments-differ
    def sample_posterior(self, n_samples, n_simulations_per_theta, K, h):
        """
        Sample from the approximate posterior

        Parameters
        ----------
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

        thetas = None
        n = n_samples
        K = jnp.maximum(
            K, self.kernel_fn(jnp.zeros((1, 2, 2)), jnp.zeros((1, 2, 2)))[0]
        )
        while n > 0:
            n_sim = jnp.minimum(n, 1000)
            ps = self.prior_sampler_fn(
                seed=next(self._rng_seq), sample_shape=(n_sim,)
            )
            ys = self.simulator_fn(
                seed=next(self._rng_seq),
                theta=jnp.tile(ps, [n_simulations_per_theta, 1, 1]),
            )
            ys = jnp.swapaxes(ys, 1, 0)
            k = self.kernel_fn(self.summary_fn(ys), self.summarized_observed, h)
            p = random.uniform(next(self._rng_seq), shape=(len(k),))
            mr = k / K
            idxs = jnp.where(p < mr)[0]
            if thetas is None:
                thetas = ps[idxs]
            else:
                thetas = jnp.vstack([thetas, ps[idxs]])
            n -= len(idxs)
        return thetas[:n_samples, :]
