import chex
import haiku as hk
from jax import numpy as jnp
from jax import random


# pylint: disable=too-many-instance-attributes
class SBI:
    """
    SBI base class
    """

    def __init__(self, model_fns):
        self.prior_sampler_fn, self.prior_log_density_fn = model_fns[0]
        self.simulator_fn = model_fns[1]
        self._len_theta = len(self.prior_sampler_fn(seed=random.PRNGKey(0)))
        self.observed: chex.Array
        self._rng_seq: hk.PRNGSequence

    def fit(self, rng_key, observed):
        """
        Fit the model

        Parameters
        ----------
        rng_seq: hk.PRNGSequence
            a hk.PRNGSequence
        observed: chex.Array
            (n \times p)-dimensional array of observations, where `n` is the n
            number of samples
        """

        self._rng_seq = hk.PRNGSequence(rng_key)
        self.observed = jnp.atleast_2d(observed)

    def sample_posterior(self, **kwargs):
        """Sample from the posterior"""
