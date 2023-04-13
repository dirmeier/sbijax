from collections import namedtuple
from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp

from sbijax._sne_base import SNE


# pylint: disable=too-many-arguments,unused-argument
class SNP(SNE):
    """
    Sequential neural posterior estimation

    From the Greenberg paper
    """

    # pylint: disable=arguments-differ,too-many-locals
    def fit(
        self,
        rng_key,
        observed,
        optimizer,
        n_rounds=10,
        n_simulations_per_round=1000,
        n_atoms=10,
        max_n_iter=1000,
        batch_size=128,
        percentage_data_as_validation_set=0.05,
        n_early_stopping_patience=10,
        **kwargs,
    ):
        """
        Fit an SNPE model

        Parameters
        ----------
        rng_seq: hk.PRNGSequence
            a hk.PRNGSequence
        observed: chex.Array
            (n \times p)-dimensional array of observations, where `n` is the n
            number of samples
        optimizer: optax.Optimizer
            an optax optimizer object
        n_rounds: int
            number of rounds to optimize
        n_simulations_per_round: int
            number of data simulations per round
        n_atoms : int
            number of atoms to approximate the proposal posterior
        max_n_iter:
            maximal number of training iterations per round
        batch_size: int
            batch size used for training the model
        percentage_data_as_validation_set:
            percentage of the simulated data that is used for valitation and
            early stopping
        n_early_stopping_patience: int
            number of iterations of no improvement of training the flow
            before stopping optimisation

        Returns
        -------
        Tuple[pytree, Tuple]
            returns a tuple of parameters and a tuple of the training
            information
        """

        super().fit(rng_key, observed)

        simulator_fn = partial(
            self._simulate_new_data_and_append,
            n_simulations_per_round=n_simulations_per_round,
        )
        D, params, all_losses, all_params = None, None, [], []
        for i_round in range(n_rounds):
            D, _ = simulator_fn(params, D, **kwargs)
            self._train_iter, self._val_iter = self.as_iterators(
                D, batch_size, percentage_data_as_validation_set
            )
            params, losses = self._fit_model_single_round(
                optimizer=optimizer,
                max_n_iter=max_n_iter,
                n_early_stopping_patience=n_early_stopping_patience,
                n_round=i_round,
                n_atoms=n_atoms,
            )
            all_params.append(params.copy())
            all_losses.append(losses)

        snp_info = namedtuple("snl_info", "params losses")
        return params, snp_info(all_params, all_losses)

    def sample_posterior(self, params, n_samples, **kwargs):
        """
        Sample from the approximate posterior

        Parameters
        ----------
        params: pytree
            a pytree of parameter for the model
        n_samples: int
            number of samples per chain

        Returns
        -------
        chex.Array
            an array of samples from the posterior distribution of dimension
            (n_samples \times p)
        """
        thetas = None
        n_curr = n_samples
        n_total_simulations_round = 0
        while n_curr > 0:
            n_sim = jnp.maximum(100, n_curr)
            n_total_simulations_round += n_sim
            proposal = self.model.apply(
                params,
                next(self.rng_seq),
                method="sample",
                sample_shape=(n_sim,),
                x=jnp.tile(self.observed, [n_sim, 1]),
            )
            proposal_probs = self.prior_log_density_fn(proposal)
            proposal_accepted = proposal[jnp.isfinite(proposal_probs)]
            if thetas is None:
                thetas = proposal_accepted
            else:
                thetas = jnp.vstack([thetas, proposal_accepted])
            n_curr -= proposal_accepted.shape[0]
        self.n_total_simulations += n_total_simulations_round
        return thetas[:n_samples], thetas.shape[0] / n_total_simulations_round

    def _fit_model_single_round(
        self, optimizer, max_n_iter, n_early_stopping_patience, n_round, n_atoms
    ):
        params = self._init_params(next(self._rng_seq), **self._train_iter(0))
        state = optimizer.init(params)

        if n_round == 0:

            def loss_fn(params, rng, **batch):
                lp = self.model.apply(
                    params,
                    None,
                    method="log_prob",
                    y=batch["theta"],
                    x=batch["y"],
                )
                return -jnp.sum(lp)

        else:

            def loss_fn(params, rng, **batch):
                lp = self._proposal_posterior_log_prob(
                    params,
                    rng,
                    n_atoms,
                    theta=batch["theta"],
                    y=batch["y"],
                )
                return -jnp.sum(lp)

        @jax.jit
        def step(params, rng, state, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, rng, **batch)
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
                batch_loss, params, state = step(
                    params, next(self.rng_seq), state, **batch
                )
                train_loss += batch_loss
            validation_loss = self._validation_loss(
                params, next(self.rng_seq), n_round, n_atoms
            )
            losses[i] = jnp.array([train_loss, validation_loss])

            _, early_stop = early_stop.update(validation_loss)
            if early_stop.should_stop:
                logging.info("early stopping criterion found")
                break

        losses = jnp.vstack(losses)[:i, :]
        return params, losses

    def _init_params(self, rng_key, **init_data):
        params = self.model.init(
            rng_key, method="log_prob", y=init_data["theta"], x=init_data["y"]
        )
        return params

    def _proposal_posterior_log_prob(self, params, rng, n_atoms, theta, y):
        n = theta.shape[0]
        n_atoms = np.maximum(2, np.minimum(n_atoms, n))
        repeated_y = jnp.repeat(y, n_atoms, axis=0)
        probs = jnp.ones((n, n)) * (1 - jnp.eye(n)) / (n - 1)

        choice = partial(
            random.choice, a=jnp.arange(n), replace=False, shape=(n_atoms - 1,)
        )
        sample_keys = random.split(rng, probs.shape[0])
        choices = jax.vmap(lambda key, prob: choice(key, p=prob))(
            sample_keys, probs
        )
        contrasting_theta = theta[choices]

        atomic_theta = jnp.concatenate(
            (theta[:, None, :], contrasting_theta), axis=1
        )
        atomic_theta = atomic_theta.reshape(n * n_atoms, -1)

        log_prob_posterior = self.model.apply(
            params, None, method="log_prob", y=atomic_theta, x=repeated_y
        )
        log_prob_posterior = log_prob_posterior.reshape(n, n_atoms)
        log_prob_prior = self.prior_log_density_fn(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(n, n_atoms)

        unnormalized_log_prob = log_prob_posterior - log_prob_prior
        log_prob_proposal_posterior = unnormalized_log_prob[
            :, 0
        ] - jsp.special.logsumexp(unnormalized_log_prob, axis=-1)

        return log_prob_proposal_posterior

    def _validation_loss(self, params, seed, n_round, n_atoms):
        if n_round == 0:

            def loss_fn(rng, **batch):
                lp = self.model.apply(
                    params,
                    None,
                    method="log_prob",
                    y=batch["theta"],
                    x=batch["y"],
                )
                return -jnp.sum(lp)

        else:

            def loss_fn(rng, **batch):
                lp = self._proposal_posterior_log_prob(
                    params, rng, n_atoms, batch["theta"], batch["y"]
                )
                return -jnp.sum(lp)

        loss = 0
        for j in range(self._val_iter.num_batches):
            rng, seed = random.split(seed)
            loss += jax.jit(loss_fn)(rng, **self._val_iter(j))
        return loss
