from collections import namedtuple
from functools import partial
from typing import Iterable

import blackjax as bj
import chex
import haiku as hk
import jax
import numpy as np
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from jax import numpy as jnp
from jax import random

from sbi import generator
from sbi.generator import named_dataset


class SNL:
    def __init__(self, model_fns, density_estimator):
        self.prior_sampler_fn, self.prior_log_density_fn = model_fns[0]()
        self.simulator_fn = model_fns[1]
        self.model = density_estimator
        self._len_theta = len(self.prior_sampler_fn(seed=random.PRNGKey(0)))

        self.observed: chex.Array
        self._rng_seq: hk.PRNGSequence
        self._train_iter: Iterable
        self._val_iter: Iterable

    def train(
        self,
        rng_key,
        observed,
        optimizer,
        n_rounds=10,
        n_simulations_per_round=1000,
        max_n_iter=1000,
        batch_size=128,
        percentage_data_as_validation_set=0.05,
        n_samples=10000,
        n_warmup=5000,
        n_chains=4,
    ):
        self._rng_seq = hk.PRNGSequence(rng_key)
        self.observed = observed

        simulator_fn = partial(
            self._simulate_new_data_and_append,
            n_simulations_per_round=n_simulations_per_round,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
        )
        D, params, all_diagnostics, all_losses = None, None, [], []
        for i in range(n_rounds):
            D, diagnostics = simulator_fn(params, D)
            self._train_iter, self._val_iter = generator.as_batch_iterators(
                next(self._rng_seq),
                D,
                batch_size,
                1.0 - percentage_data_as_validation_set,
                True,
            )
            params, losses = self._train_model_single_round(
                optimizer, max_n_iter
            )
            all_losses.append(losses)
            all_diagnostics.append(diagnostics)

        return params, all_losses, all_diagnostics

    def _train_model_single_round(self, optimizer, max_n_iter):
        params = self._init_params(next(self._rng_seq), self._train_iter(0))
        state = optimizer.init(params)

        @jax.jit
        def step(params, rng, state, **batch):
            def loss_fn(params):
                lp = self.model.apply(params, method="log_prob", **batch)
                return -jnp.sum(lp)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([max_n_iter, 2])
        early_stop = EarlyStopping(1e-3, 5)
        for i in range(max_n_iter):
            train_loss = 0.0
            for j in range(self._train_iter.num_batches):
                batch = self._train_iter(j)
                batch_loss, params, state = step(
                    params, next(self._rng_seq), state, **batch
                )
                train_loss += batch_loss
            validation_loss = self._validation_loss(params)
            losses[i] = jnp.array([train_loss, validation_loss])

            _, early_stop = early_stop.update(validation_loss)
            if early_stop.should_stop:
                logging.info("early stop criterion found")
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

    def _init_params(self, rng_key, init_data):
        params = self.model.init(rng_key, method="log_prob", **init_data)
        return params

    def _simulate_new_data_and_append(
        self, params, D, n_simulations_per_round, n_chains, n_samples, n_warmup
    ):
        if D is None:
            diagnostics = None
            new_thetas = self.prior_sampler_fn(
                seed=next(self._rng_seq),
                sample_shape=(n_simulations_per_round,),
            )
        else:
            new_thetas, diagnostics = self._simulate_from_amortized_posterior(
                params, n_chains, n_samples, n_warmup
            )
            new_thetas = random.permutation(next(self._rng_seq), new_thetas)
            new_thetas = new_thetas[:n_simulations_per_round, :]

        new_data = self.simulator_fn(seed=next(self._rng_seq), theta=new_thetas)
        if D is None:
            d_new = named_dataset(new_data, new_thetas)
        else:
            d_new = named_dataset(
                *[jnp.vstack([a, b]) for a, b in zip(D, new_data)]
            )
        return d_new, diagnostics

    def _simulate_from_amortized_posterior(
        self, params, n_chains, n_samples, n_warmup
    ):
        part = partial(
            self.model.apply, params=params, method="log_prob", y=self.observed
        )

        def _log_likelihood_fn(theta):
            theta = jnp.tile(theta, [self.observed.shape[0], 1])
            return part(x=theta)

        def _joint_logdensity_fn(theta):
            lp_prior = self.prior_log_density_fn(theta)
            lp = _log_likelihood_fn(theta)
            return jnp.sum(lp) + jnp.sum(lp_prior)

        lp__ = lambda x: _joint_logdensity_fn(**x)

        def _inference_loop(rng_key, kernel, initial_state, n_samples):
            @jax.jit
            def _step(states, rng_key):
                keys = jax.random.split(rng_key, n_chains)
                states, infos = jax.vmap(kernel)(keys, states)
                return states, states

            sampling_keys = jax.random.split(rng_key, n_samples)
            _, states = jax.lax.scan(_step, initial_state, sampling_keys)
            return states

        initial_positions = random.multivariate_normal(
            next(self._rng_seq),
            0.1 + jnp.zeros(self._len_theta),
            jnp.eye(self._len_theta),
            shape=(n_chains,),
        )
        initial_positions = {"theta": initial_positions}

        init_keys = random.split(next(self._rng_seq), n_chains)
        warmup = bj.window_adaptation(bj.nuts, lp__)
        initial_states, kernel_params = jax.vmap(
            lambda seed, param: warmup.run(seed, param)[0]
        )(init_keys, initial_positions)

        kernel_params = {k: v[0] for k, v in kernel_params.items()}
        _, kernel = bj.nuts(lp__, **kernel_params)

        states = _inference_loop(
            next(self._rng_seq), kernel, initial_states, n_samples
        )
        _ = states.position["theta"].block_until_ready()
        thetas = states.position["theta"][n_warmup:, :, :]

        thetas, diagnostics = self._merge_chains_and_diagnose(thetas)
        return thetas, diagnostics

    def _merge_chains_and_diagnose(self, samples):
        esses = [0] * samples.shape[-1]
        rhats = [0] * samples.shape[-1]
        for i in range(samples.shape[-1]):
            esses[i] = bj.diagnostics.effective_sample_size(samples[:, :, i].T)
            rhats[i] = bj.diagnostics.potential_scale_reduction(
                samples[:, :, i].T
            )
        samples = samples.reshape(-1, samples.shape[-1])
        return samples, namedtuple("diagnostics", "ess rhat")(esses, rhats)
