import abc
from abc import ABC

import chex
from jax import numpy as jnp
from jax import random as jr

from sbijax._src._sbi_base import SBI
from sbijax._src.util.data import stack_data
from sbijax._src.util.dataloader import as_batch_iterators, named_dataset


# ruff: noqa: PLR0913
class SNE(SBI, ABC):
    """Sequential neural estimation base class."""

    def __init__(self, model_fns, network):
        """Construct an SNE object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            network: a neural network
        """
        super().__init__(model_fns)
        self.model = network
        self.n_total_simulations = 0

    def simulate_data_and_possibly_append(
        self,
        rng_key,
        params,
        observable,
        data=None,
        n_simulations=1000,
        **kwargs,
    ):
        """Simulate data from the  prior or posterior and append.

        Args:
            rng_key: a random key
            params: a dictionary of neural network parameters
            observable: an observation
            data: existing data set
            n_simulations: number of newly simulated data
            kwargs: dictionary of ey value pairs passed to `sample_posterior`

        Returns:
            returns a NamedTuple of two axis, y and theta
        """
        observable = jnp.atleast_2d(observable)
        new_data, diagnostics = self.simulate_data(
            rng_key,
            params=params,
            observable=observable,
            n_simulations=n_simulations,
            **kwargs,
        )
        if data is None:
            d_new = new_data
        else:
            d_new = stack_data(data, new_data)
        return d_new, diagnostics

    @abc.abstractmethod
    def sample_posterior(self, rng_key, params, observable, *args, **kwargs):
        """Sample from the approximate posterior.

        Args:
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: a data point
            *args: argument list
            **kwargs: keyword arguments
        """

    def simulate_data(
        self,
        rng_key,
        *,
        params=None,
        observable=None,
        n_simulations=1000,
        **kwargs,
    ):
        r"""Simulate data from the posterior or prior and append.

        Args:
            rng_key: a random key
            params:a dictionary of neural network parameters. If None, will
                draw from prior. If parameters given, will draw from amortized
                posterior using 'observable;
            observable: an observation. Needs to be gfiven if posterior draws
                are desired
            n_simulations: number of newly simulated data
            kwargs: dictionary of ey value pairs passed to `sample_posterior`

        Returns:
            a NamedTuple of two axis, y and theta
        """
        sample_key, rng_key = jr.split(rng_key)
        if params is None or len(params) == 0:
            diagnostics = None
            self.n_total_simulations += n_simulations
            new_thetas = self.prior_sampler_fn(
                seed=sample_key,
                sample_shape=(n_simulations,),
            )
        else:
            if observable is None:
                raise ValueError(
                    "need to have access to 'observable' "
                    "when sampling from posterior"
                )
            if "n_samples" not in kwargs:
                kwargs["n_samples"] = n_simulations
            new_thetas, diagnostics = self.sample_posterior(
                rng_key=sample_key,
                params=params,
                observable=jnp.atleast_2d(observable),
                **kwargs,
            )
            perm_key, rng_key = jr.split(rng_key)
            new_thetas = jr.permutation(perm_key, new_thetas)
            new_thetas = new_thetas[:n_simulations, :]

        simulate_key, rng_key = jr.split(rng_key)
        new_obs = self.simulator_fn(seed=simulate_key, theta=new_thetas)
        chex.assert_shape(new_thetas, [n_simulations, None])
        chex.assert_shape(new_obs, [n_simulations, None])

        new_data = named_dataset(new_obs, new_thetas)

        return new_data, diagnostics

    @staticmethod
    def as_iterators(
        rng_key, data, batch_size, percentage_data_as_validation_set
    ):
        """Convert the data set to an iterable for training.

        Args:
            rng_key: a jax random key
            data: a tuple with 'y' and 'theta' elements
            batch_size: the size of each batch
            percentage_data_as_validation_set: fraction

        Returns:
            two batch iterators
        """
        return as_batch_iterators(
            rng_key,
            data,
            batch_size,
            1.0 - percentage_data_as_validation_set,
            True,
        )
