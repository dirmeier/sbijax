import abc
from typing import Optional

import chex
import haiku as hk
from jax import numpy as jnp
from jax import random

from sbijax.generator import named_dataset


# pylint: disable=too-many-instance-attributes
class SBI(abc.ABC):
    """
    SBI base class
    """

    def __init__(self, model_fns):
        self.prior_sampler_fn, self.prior_log_density_fn = model_fns[0]
        self.simulator_fn = model_fns[1]
        self._len_theta = len(self.prior_sampler_fn(seed=random.PRNGKey(0)))

        self._observed: chex.Array
        self._rng_seq: hk.PRNGSequence
        self._data: Optional[named_dataset] = None

    @property
    def observed(self):
        return self._observed

    @observed.setter
    def observed(self, observed):
        self._observed = jnp.atleast_2d(observed)

    @property
    def data(self):
        return self._data

    @property
    def rng_seq(self):
        return self._rng_seq

    @rng_seq.setter
    def rng_seq(self, rng_seq):
        self._rng_seq = rng_seq

    def fit(self, rng_key, observed, **kwargs):
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

        self.rng_seq = hk.PRNGSequence(rng_key)
        self.observed = observed

    @abc.abstractmethod
    def sample_posterior(self, **kwargs):
        """Sample from the posterior"""
