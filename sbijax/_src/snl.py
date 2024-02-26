from functools import partial

import chex
import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from sbijax._src import mcmc
from sbijax._src._sne_base import SNE
from sbijax._src.mcmc import mcmc_diagnostics
from sbijax._src.util.early_stopping import EarlyStopping


# pylint: disable=too-many-arguments,unused-argument
class SNL(SNE):
    """Sequential neural likelihood.

    Implements both SNL and SSNL estimation methods.

    Args:
        model_fns: a tuple of tuples. The first element is a tuple that
                consists of functions to sample and evaluate the
                log-probability of a data point. The second element is a
                simulator function.
        density_estimator: a (neural) conditional density estimator
            to model the likelihood function

    References:
        .. [1] Papamakarios, George, et al. "Sequential neural likelihood:
           Fast likelihood-free inference with autoregressive flows."
           International Conference on Artificial Intelligence and Statistics,
           2019.
        .. [2] Dirmeier, Simon, et al. "Simulation-based inference using
           surjective sequential neural likelihood estimation."
           arXiv preprint arXiv:2308.01054, 2023.
    """

    # pylint: disable=useless-parent-delegation
    def __init__(self, model_fns, density_estimator):
        """Construct a SNL object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            density_estimator: a (neural) conditional density estimator
                        to model the likelihood function
        """
        super().__init__(model_fns, density_estimator)

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
        """Fit a SNL or SSNL model.

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
        params = self._init_params(init_key, **next(iter(train_iter)))
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
                params, method="log_prob", y=batch["y"], x=batch["theta"]
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

            def lp__(theta):
                return jax.vmap(_joint_logdensity_fn)(theta)

            sampler = kwargs.pop("sampler", None)
        else:

            def lp__(theta):
                return _joint_logdensity_fn(**theta)

            # take whatever sampler is or per default nuts
            sampler = kwargs.pop("sampler", "nuts")

        sampling_fn = getattr(mcmc, "sample_with_" + sampler)
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
