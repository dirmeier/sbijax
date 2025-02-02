from functools import partial

import arviz
import chex
import jax
import numpy as np
import optax
from absl import logging
from arviz import InferenceData
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree
from tqdm import tqdm

from sbijax._src import mcmc
from sbijax._src._ne_base import NE
from sbijax._src.mcmc.util import mcmc_diagnostics
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.early_stopping import EarlyStopping


# ruff: noqa: PLR0913, E501
class NLE(NE):
    """Neural likelihood estimation.

    Implements the method introduced in :cite:t:`papama2019neural`.

    Args:
        model_fns: a tuple of calalbles. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        density_estimator: a (neural) conditional density estimator
            to model the likelihood function

    Examples:
        >>> from sbijax import NLE
        >>> from sbijax.nn import make_mdn
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...    dict(theta=tfd.Normal(0.0, 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> neural_network = make_mdn(1, 5)
        >>> model = NLE(fns, neural_network)

    References:
        Papamakarios, George, et al. "Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows." International Conference on Artificial Intelligence and Statistics, 2019.
    """

    # pylint: disable=arguments-differ,too-many-locals
    def fit(
        self,
        rng_key,
        data,
        optimizer=optax.adam(0.0003),
        n_iter=1000,
        batch_size=100,
        percentage_data_as_validation_set=0.1,
        n_early_stopping_patience=10,
        **kwargs,
    ):
        """Fit the model.

        Args:
            rng_key: a jax random key
            data: data set obtained from calling
                `simulate_data_and_possibly_append`
            optimizer: an optax optimizer object
            n_iter: maximal number of training iterations per round
            batch_size: batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated data
                that is used for valitation and early stopping
            n_early_stopping_patience: number of iterations of no improvement of
                training the flow before stopping optimisation
            **kwargs: additional keyword arguments (not used for NLE)

        Returns:
            a tuple of parameters and a tuple of the training
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

    # pylint: disable=arguments-differ,undefined-loop-variable
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
        params = self._init_params(init_key, **next(iter(train_iter)))
        state = optimizer.init(params)

        @jax.jit
        def step(params, state, **batch):
            def loss_fn(params):
                lp = self.model.apply(
                    params,
                    rng=None,
                    method="log_prob",
                    y=batch["y"],
                    x=batch["theta"],
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
        for i in tqdm(range(n_iter)):
            train_loss = 0.0
            for batch in train_iter:
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
                params,
                rng=None,
                method="log_prob",
                y=batch["y"],
                x=batch["theta"],
            )
            return -jnp.mean(lp)

        def body_fn(batch):
            loss = loss_fn(**batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        losses = 0.0
        for batch in val_iter:
            losses += body_fn(batch)
        return losses

    def _init_params(self, rng_key, **init_data):
        params = self.model.init(
            rng_key, method="log_prob", y=init_data["y"], x=init_data["theta"]
        )
        return params

    # ruff: noqa: D417
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
            n_warmup: number of draws to discarded

        Keyword Args:
            sampler (str): either 'nuts', 'slice' or None (defaults to nuts)
            n_thin (int): number of thinning steps
                (only used if sampler='slice')
            n_doubling (int): number of doubling steps of the interval
                 (only used if sampler='slice')
            step_size (float): step size of the initial interval
                 (only used if sampler='slice')

        Returns:
            returns a NamedTuple with two elements, y and theta
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

    def simulate_data(
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
        return super().simulate_data(
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
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: observation to condition on
            n_chains: number of MCMC chains
            n_samples: number of samples per chain
            n_warmup:  number of samples to discard

        Keyword Args:
            sampler (str): either 'nuts', 'slice' or None (defaults to nuts)
            n_thin (int): number of thinning steps
                (only used if sampler='slice')
            n_doubling (int): number of doubling steps of the interval
                 (only used if sampler='slice')
            step_size (float): step size of the initial interval
                 (only used if sampler='slice')

        Returns:
            an array of samples from the posterior distribution of dimension
            (n_samples \times p) and posterior diagnostics
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
            self.model.apply,
            params=params,
            rng=None,
            method="log_prob",
            y=observable,
        )

        def _log_likelihood_fn(theta):
            theta, _ = ravel_pytree(theta)
            theta = jnp.tile(theta, [observable.shape[0], 1])
            return part(x=theta)

        def _prop_posterior_density(theta):
            lp_prior = self.prior.log_prob(theta)
            lp = _log_likelihood_fn(theta)
            return jnp.sum(lp) + jnp.sum(lp_prior)

        sampler = kwargs.pop("sampler", "nuts")
        sampling_fn = getattr(mcmc, "sample_with_" + sampler)
        samples = sampling_fn(
            rng_key=rng_key,
            lp=_prop_posterior_density,
            prior=self.prior,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )
        for v in samples.values():
            chex.assert_shape(v, [n_chains, n_samples - n_warmup, None])
        inference_data = as_inference_data(samples, jnp.squeeze(observable))
        diagnostics = mcmc_diagnostics(inference_data)
        return inference_data, diagnostics

    def _simulate_parameters_with_model(
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
        return self.sample_posterior(
            rng_key=rng_key,
            params=params,
            observable=observable,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )

    @staticmethod
    def plot(inference_data: InferenceData):
        arviz.plot_trace(inference_data)
