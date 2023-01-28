import chex
import haiku as hk
from jax import numpy as jnp
from jax import random


class RejectionABC:
    """
    Sisson et al. - Handbook of approximate Bayesian computation

    Algorithm 4.1, "ABC Rejection Sampling Algorithm"
    """

    def __init__(self, model_fns, summary_fn, kernel_fn):
        self.prior_sampler_fn, self.prior_log_density_fn = model_fns[0]
        self.simulator_fn = model_fns[1]
        self.summary_fn = summary_fn
        self.kernel_fn = kernel_fn
        self._len_theta = len(self.prior_sampler_fn(seed=random.PRNGKey(0)))

        self.observed: chex.Array
        self.summarized_observed: chex.Array
        self._rng_seq: hk.PRNGSequence

    def fit(self, rng_key, observed):
        self._rng_seq = hk.PRNGSequence(rng_key)
        self.observed = jnp.atleast_2d(observed)
        self.summarized_observed = self.summary_fn(self.observed)

    def sample_posterior(self, n_samples, n_simulations_per_theta, K):
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
            k = self.kernel_fn(self.summary_fn(ys), self.summarized_observed)
            p = random.uniform(next(self._rng_seq), shape=(len(k),))
            mr = k / K
            idxs = jnp.where(p < mr)[0]
            if thetas is None:
                thetas = ps[idxs]
            else:
                thetas = jnp.vstack([thetas, ps[idxs]])
            n -= len(idxs)
        return thetas[:n_samples, :]
