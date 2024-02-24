from functools import partial

import chex
import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr

from sbijax._src._sne_base import SNE
from sbijax._src.mcmc import (
    mcmc_diagnostics,
    sample_with_nuts,
    sample_with_slice,
)
from sbijax._src.mcmc.irmh import sample_with_imh
from sbijax._src.mcmc.mala import sample_with_mala
from sbijax._src.mcmc.rmh import sample_with_rmh
from sbijax._src.util.early_stopping import EarlyStopping


# pylint: disable=too-many-arguments,unused-argument
class SNL(SNE):
    """Sequential neural likelihood.

    Implements SNL and SSNL estimation methods.

    References:
        .. [1] Papamakarios, George, et al. "Sequential neural likelihood:
           Fast likelihood-free inference with autoregressive flows."
           International Conference on Artificial Intelligence and Statistics,
           2019.
        .. [2] Dirmeier, Simon, et al. "Simulation-based inference using
           surjective sequential neural likelihood estimation."
           arXiv preprint arXiv:2308.01054, 2023.
    """

    # pylint: disable=arguments-differ,too-many-locals
    def fit(
        self,
        rng_key,
        data,
        optimizer=optax.adam(0.0003),
        n_iter=1000,
        batch_size=128,
        percentage_data_as_validation_set=0.1,
        n_early_stopping_patience=10,
        **kwargs,
    ):
        """Fit a SNL model.

        Args:
            rng_key: a hk.PRNGSequence
            data: data set obtained from calling
                `simulate_data_and_possibly_append`
            optimizer: an optax optimizer object
            n_iter: maximal number of training iterations per round
            batch_size: batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated data
                that is used for valitation and early stopping
            n_early_stopping_patience: number of iterations of no improvement of
                training the flow before stopping optimisation
            kwargs: keyword arguments with sampler specific parameters.
                For slice sampling the following arguments are possible:
                - sampler: either 'nuts', 'slice' or None (defaults to nuts)
                - n_thin: number of thinning steps
                - n_doubling: number of doubling steps of the interval
                - step_size: step size of the initial interval

        Returns:
            returns a tuple of parameters and a tuple of the training
            information
        """
        itr_key, rng_key = jr.split(rng_key)
        train_iter, val_iter = self.as_iterators(
            itr_key, data, batch_size, percentage_data_as_validation_set
        )
        params, losses = self._fit_model_single_round(
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
        )

        return params, losses

    # pylint: disable=arguments-differ
    def _fit_model_single_round(
        self,
        seed,
        train_iter,
        val_iter,
        optimizer,
        n_iter,
        n_early_stopping_patience,
    ):
        init_key, seed = jr.split(seed)
        params = self._init_params(init_key, **train_iter(0))
        state = optimizer.init(params)

        @jax.jit
        def step(params, state, **batch):
            def loss_fn(params):
                lp = self.model.apply(
                    params, method="log_prob", y=batch["y"], x=batch["theta"]
                )
                return -jnp.mean(lp)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
        best_params, best_loss = None, np.inf
        logging.info("training model")
        for i in range(n_iter):
            train_loss = 0.0
            for j in range(train_iter.num_batches):
                batch = train_iter(j)
                batch_loss, params, state = step(params, state, **batch)
                train_loss += batch_loss * (
                    batch["y"].shape[0] / train_iter.num_samples
                )
            validation_loss = self._validation_loss(params, val_iter)
            losses[i] = jnp.array([train_loss, validation_loss])

            _, early_stop = early_stop.update(validation_loss)
            if early_stop.should_stop:
                logging.info("early stopping criterion found")
                break
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_params = params.copy()

        losses = jnp.vstack(losses)[: (i + 1), :]
        return best_params, losses

    def _validation_loss(self, params, val_iter):
        @jax.jit
        def loss_fn(**batch):
            lp = self.model.apply(
                params, method="log_prob", y=batch["y"], x=batch["theta"]
            )
            return -jnp.mean(lp)

        def body_fn(i):
            batch = val_iter(i)
            loss = loss_fn(**batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        losses = 0.0
        for i in range(val_iter.num_batches):
            losses += body_fn(i)
        return losses

    def _init_params(self, rng_key, **init_data):
        params = self.model.init(
            rng_key, method="log_prob", y=init_data["y"], x=init_data["theta"]
        )
        return params

    def simulate_data_and_possibly_append(
        self,
        rng_key,
        params=None,
        observable=None,
        data=None,
        n_simulations=1_000,
        n_chains=4,
        n_samples=2_000,
        n_warmup=1_000,
        **kwargs,
    ):
        """Simulate data from the prior or posterior.

        Args:
            rng_key: a random key
            params: a dictionary of neural network parameters
            observable: an observation
            data: existing data set
            n_simulations: number of newly simulated data
            n_chains: number of MCMC chains
            n_samples: number of sa les to draw in total
            n_warmup: number of draws to discared
            kwargs: keyword arguments
               dictionary of ey value pairs passed to `sample_posterior`.
               The following arguments are possible:
               - sampler: either 'nuts', 'slice' or None (defaults to nuts)
               - n_thin: number of thinning steps (int)
               - n_doubling: number of doubling steps of the interval (int)
               - step_size: step size of the initial interval (float)

        Returns:
           returns a NamedTuple of two axis, y and theta
        """
        return super().simulate_data_and_possibly_append(
            rng_key=rng_key,
            params=params,
            observable=observable,
            data=data,
            n_simulations=n_simulations,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )

    def sample_posterior(
        self,
        rng_key,
        params,
        observable,
        *,
        n_chains=4,
        n_samples=2_000,
        n_warmup=1_000,
        **kwargs,
    ):
        r"""Sample from the approximate posterior.

        Args:
            rng_key: a random key
            params: a pytree of parameter for the model
            observable: observation to condition on
            n_chains: number of MCMC chains
            n_samples: number of samples per chain
            n_warmup:  number of samples to discard
            kwargs: keyword arguments with sampler specific parameters. For
                slice sampling the following arguments are possible:
                - sampler: either 'nuts', 'slice' or None (defaults to nuts)
                - n_thin: number of thinning steps
                - n_doubling: number of doubling steps of the interval
                - step_size: step size of the initial interval

        Returns:
            an array of samples from the posterior distribution of dimension
            (n_samples \times p)
        """
        observable = jnp.atleast_2d(observable)
        return self._sample_posterior(
            rng_key,
            params,
            observable,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )

    def _sample_posterior(
        self,
        rng_key,
        params,
        observable,
        *,
        n_chains=4,
        n_samples=2_000,
        n_warmup=1_000,
        **kwargs,
    ):

        part = partial(
            self.model.apply, params=params, method="log_prob", y=observable
        )

        def _log_likelihood_fn(theta):
            theta = jnp.tile(theta, [observable.shape[0], 1])
            return part(x=theta)

        def _joint_logdensity_fn(theta):
            lp_prior = self.prior_log_density_fn(theta)
            lp = _log_likelihood_fn(theta)
            return jnp.sum(lp) + jnp.sum(lp_prior)

        if "sampler" in kwargs and kwargs["sampler"] == "slice":
            kwargs.pop("sampler", None)

            def lp__(theta):
                return jax.vmap(_joint_logdensity_fn)(theta)

            sampling_fn = sample_with_slice
        elif "sampler" in kwargs and kwargs["sampler"] == "rmh":
            kwargs.pop("sampler", None)

            def lp__(theta):
                return _joint_logdensity_fn(**theta)

            sampling_fn = sample_with_rmh
        elif "sampler" in kwargs and kwargs["sampler"] == "imh":
            kwargs.pop("sampler", None)

            def lp__(theta):
                return _joint_logdensity_fn(**theta)

            sampling_fn = sample_with_imh
        elif "sampler" in kwargs and kwargs["sampler"] == "mala":
            kwargs.pop("sampler", None)

            def lp__(theta):
                return _joint_logdensity_fn(**theta)

            sampling_fn = sample_with_mala
        else:

            def lp__(theta):
                return _joint_logdensity_fn(**theta)

            sampling_fn = sample_with_nuts

        samples = sampling_fn(
            rng_key=rng_key,
            lp=lp__,
            prior=self.prior_sampler_fn,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )
        chex.assert_shape(samples, [n_samples - n_warmup, n_chains, None])
        diagnostics = mcmc_diagnostics(samples)
        samples = samples.reshape((n_samples - n_warmup) * n_chains, -1)

        return samples, diagnostics