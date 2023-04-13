from collections import namedtuple
from functools import partial
from typing import Iterable

import chex
import haiku as hk
import jax
import numpy as np
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from jax import numpy as jnp
from jax import random

from sbijax import generator
from sbijax._sbi_base import SBI
from sbijax.generator import named_dataset
from sbijax.mcmc import sample_with_nuts
from sbijax.mcmc.sample import mcmc_diagnostics

# pylint: disable=too-many-arguments
from sbijax.mcmc.slice import sample_with_slice


class SNE(SBI):
    """
    Sequential neural estimation
    """

    def __init__(self, model_fns, density_estimator):
        super().__init__(model_fns)
        self.model = density_estimator
        self._len_theta = len(self.prior_sampler_fn(seed=random.PRNGKey(0)))

        self._train_iter: Iterable
        self._val_iter: Iterable

    def _fit_model_single_round(
        self, optimizer, max_n_iter, n_early_stopping_patience
    ):
        params = self._init_params(next(self._rng_seq), self._train_iter(0))
        state = optimizer.init(params)

        @jax.jit
        def step(params, state, **batch):
            def loss_fn(params):
                lp = self.model.apply(params, method="log_prob", **batch)
                return -jnp.sum(lp)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([max_n_iter, 2])
        early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
        logging.info("training model")
        for i in range(max_n_iter):
            train_loss = 0.0
            for j in range(self._train_iter.num_batches):
                batch = self._train_iter(j)
                batch_loss, params, state = step(params, state, **batch)
                train_loss += batch_loss
            validation_loss = self._validation_loss(params)
            losses[i] = jnp.array([train_loss, validation_loss])

            _, early_stop = early_stop.update(validation_loss)
            if early_stop.should_stop:
                logging.info("early stopping criterion found")
                break

        losses = jnp.vstack(losses)[:i, :]
        return params, losses

    def _validation_loss(self, params):
        def _loss_fn(**batch):
            lp = self.model.apply(params, method="log_prob", **batch)
            return -jnp.sum(lp)

        losses = jnp.array(
            [
                _loss_fn(**self._val_iter(j))
                for j in range(self._val_iter.num_batches)
            ]
        )
        return jnp.sum(losses)
