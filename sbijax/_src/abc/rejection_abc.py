from typing import Callable, Tuple

from jax import numpy as jnp
from jax import random as jr

from sbijax._src._sbi_base import SBI


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-few-public-methods,
class RejectionABC(SBI):
    """Rejection approximate Bayesian computation.

    Implements algorithm~4.1 from [1].

    References:
        .. [1] Sisson, Scott A, et al. "Handbook of approximate Bayesian
           computation". 2019
    """

    def __init__(
        self, model_fns: Tuple, summary_fn: Callable, kernel_fn: Callable
    ):
        """Constructs a RejectionABC object.

        Args:
            model_fns: tuple
            summary_fn: summary statistice function
            kernel_fn: a kernel function to compute similarities
        """
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
        r"""Sample from the approximate posterior.

        Args:
            rng_key: a random key
            observable: observation to condition on
            n_samples: number of samples to draw for each parameter
            n_simulations_per_theta: number of simulations for each parameter
                sample
            K: normalisation parameter
            h: kernel scale

        Returns:
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
