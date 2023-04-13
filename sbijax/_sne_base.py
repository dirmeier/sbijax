from abc import ABC
from typing import Iterable

from jax import numpy as jnp

from sbijax._sbi_base import SBI

# pylint: disable=too-many-arguments
from sbijax.generator import named_dataset


class SNE(SBI, ABC):
    """
    Sequential neural estimation
    """

    def __init__(self, model_fns, density_estimator):
        super().__init__(model_fns)
        self.model = density_estimator

        self._train_iter: Iterable
        self._val_iter: Iterable

    def simulate_new_data_and_append(self, params, n_simulations):
        self._data = self._simulate_new_data_and_append(
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
            new_thetas = self.prior_sampler_fn(
                seed=next(self._rng_seq),
                sample_shape=(n_simulations_per_round,),
            )
        else:
            new_thetas = self.sample_posterior(
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
        return d_new
