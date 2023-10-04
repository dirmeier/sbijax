from abc import ABC
from typing import Iterable

import chex
from jax import numpy as jnp
from jax import random as jr

from sbijax import generator
from sbijax._sbi_base import SBI
from sbijax.generator import named_dataset


# pylint: disable=too-many-arguments,unused-argument
# pylint: disable=too-many-function-args,arguments-differ
class SNE(SBI, ABC):
    """
    Sequential neural estimation
    """

    def __init__(self, model_fns, density_estimator):
        super().__init__(model_fns)
        self.model = density_estimator
        self.n_total_simulations = 0
        self._train_iter: Iterable
        self._val_iter: Iterable

    def simulate_data_and_possibly_append(
        self,
        rng_key,
        params,
        observable,
        data=None,
        n_simulations=1000,
        **kwargs,
    ):
        """
        Simulate data from the posteriorand append it to an existing data set
         (if provided)

        Parameters
        ----------
        rng_key: jax.PRNGKey
            a random key
        params: pytree
            a dictionary of neural network parameters
        observable: jnp.ndarray
            an observation
        data: NamedTuple
            existing data set
        n_simulations: int
            number of newly simulated data
        kwargs: keyword arguments
            dictionary of ey value pairs passed to `sample_posterior`

        Returns
        -------
        NamedTuple:
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
            d_new = self.stack_data(data, new_data)
        return d_new, diagnostics

    def simulate_data(
        self,
        rng_key,
        *,
        params=None,
        observable=None,
        n_simulations=1000,
        **kwargs,
    ):
        """
        Simulate data from the posterior or prior and append it to an
        existing data set (if provided)

        Parameters
        ----------
        rng_key: jax.PRNGKey
            a random key
        params: Optional[pytree]
            a dictionary of neural network parameters. If None, will draw from
            prior. If parameters given, will draw from amortized posterior
            using 'observable;
        observable: Optional[jnp.ndarray]
            an observation. Needs to be gfiven if posterior draws are desired
        n_simulations: int
            number of newly simulated data
        kwargs: keyword arguments
            dictionary of ey value pairs passed to `sample_posterior`

        Returns
        -------
        NamedTuple:
            returns a NamedTuple of two axis, y and theta
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
    def stack_data(data, also_data):
        """
        Stack two data sets.

        Parameters
        ----------
        data: NamedTuple
            one data set
        also_data: : NamedTuple

        Returns
        -------
        NamedTuple:
            returns the stack of the two data sets
        """

        return named_dataset(
            *[jnp.vstack([a, b]) for a, b in zip(data, also_data)]
        )

    def as_iterators(
        self, rng_key, data, batch_size, percentage_data_as_validation_set
    ):
        """Convert the data set to an iterable for training"""
        return generator.as_batch_iterators(
            rng_key,
            data,
            batch_size,
            1.0 - percentage_data_as_validation_set,
            True,
        )
