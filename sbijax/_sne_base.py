from abc import ABC
from typing import Iterable

from jax import numpy as jnp

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

    def simulate_new_data_and_append(self, params, n_simulations):
        """
        Simulate novel data-parameters pairs and append to the
        existing data set.

        Parameters
        ----------
        params: pytree
            parameter set of the neural network
        n_simulations: int
            number of data-parameter pairs to draw

        Returns
        -------
        Returns the data set.
        """

        self.data = self._simulate_new_data_and_append(
            params, self.data, n_simulations
        )
        return self.data

    def _simulate_new_data_and_append(
        self,
        params,
        D,
        n_simulations_per_round,
        **kwargs,
    ):
        if D is None:
            diagnostics = None
            self.n_total_simulations += n_simulations_per_round
            new_thetas = self.prior_sampler_fn(
                seed=next(self._rng_seq),
                sample_shape=(n_simulations_per_round,),
            )
        else:
            new_thetas, diagnostics = self.sample_posterior(
                params, n_simulations_per_round, **kwargs
            )

        new_obs = self.simulator_fn(seed=next(self._rng_seq), theta=new_thetas)
        new_data = named_dataset(new_obs, new_thetas)
        if D is None:
            d_new = new_data
        else:
            d_new = named_dataset(
                *[jnp.vstack([a, b]) for a, b in zip(D, new_data)]
            )
        return d_new, diagnostics

    def as_iterators(self, D, batch_size, percentage_data_as_validation_set):
        """Convert the data set to an iterable for training"""
        return generator.as_batch_iterators(
            next(self._rng_seq),
            D,
            batch_size,
            1.0 - percentage_data_as_validation_set,
            True,
        )
