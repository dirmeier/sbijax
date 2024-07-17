from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree
from tqdm import tqdm

from sbijax._src._ne_base import NE
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.early_stopping import EarlyStopping
from sbijax._src.util.types import PyTree


def _sample_theta_t(rng_key, times, theta, sigma_min):
    mus = times * theta
    sigmata = 1.0 - (1.0 - sigma_min) * times
    sigmata = sigmata.reshape(times.shape[0], 1)

    noise = jr.normal(rng_key, shape=(*theta.shape,))
    theta_t = noise * sigmata + mus
    return theta_t


def _ut(theta_t, theta, times, sigma_min):
    num = theta - (1.0 - sigma_min) * theta_t
    denom = 1.0 - (1.0 - sigma_min) * times
    return num / denom


# pylint: disable=too-many-locals
def _cfm_loss(
    params, rng_key, apply_fn, sigma_min=0.001, is_training=True, **batch
):
    theta = batch["theta"]
    n, _ = theta.shape

    t_key, rng_key = jr.split(rng_key)
    times = jr.uniform(t_key, shape=(n, 1))

    theta_key, rng_key = jr.split(rng_key)
    theta_t = _sample_theta_t(theta_key, times, theta, sigma_min)

    train_rng, rng_key = jr.split(rng_key)
    vs = apply_fn(
        params,
        train_rng,
        method="vector_field",
        theta=theta_t,
        time=times,
        context=batch["y"],
        is_training=is_training,
    )
    uts = _ut(theta_t, theta, times, sigma_min)

    loss = jnp.mean(jnp.square(vs - uts))
    return loss


# ruff: noqa: PLR0913
class FMPE(NE):
    r"""Flow matching posterior estimation.

    Implements the FMPE algorithm introduced in :cite:t:`wilderberger2023flow`.

    Args:
        model_fns: a tuple of tuples. The first element is a tuple that
                consists of functions to sample and evaluate the
                log-probability of a data point. The second element is a
                simulator function.
        density_estimator: a continuous normalizing flow model

    Examples:
        >>> from sbijax import FMPE
        >>> from sbijax.nn import make_cnf
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(dict(theta=tfd.Normal(0.0, 1.0)))
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> flow = make_cnf(1)
        ...
        >>> estim = FMPE(fns, flow)

    References:
        Wildberger, Jonas, et al. "Flow Matching for Scalable Simulation-Based Inference." Advances in Neural Information Processing Systems, 2024.
    """

    def __init__(self, model_fns, density_estimator):
        super().__init__(model_fns, density_estimator)

    def fit(
        self,
        rng_key: jr.PRNGKey,
        data: PyTree,
        *,
        optimizer: optax.GradientTransformation=optax.adam(0.0003),
        n_iter:int=1000,
        batch_size:int=100,
        percentage_data_as_validation_set:float=0.1,
        n_early_stopping_patience:int=10,
        n_early_stopping_delta:float=0.001,
        **kwargs,
    ):
        """Fit the model.

        Args:
            rng_key: a jax random key
            data: data set obtained from calling
                `simulate_data_and_possibly_append`
            optimizer: an optax optimizer object
            n_iter: maximal number of training iterations per round
            batch_size:  batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated
                data that is used for validation and early stopping
            n_early_stopping_patience: number of iterations of no improvement
                of training the flow before stopping optimisation
            **kwargs: optional keyword arguments

        Returns:
            a tuple of parameters and a tuple of the training information
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
            n_early_stopping_delta=n_early_stopping_delta,
        )

        return params, losses

    def _fit_model_single_round(
        self,
        seed,
        train_iter,
        val_iter,
        optimizer,
        n_iter,
        n_early_stopping_patience,
        n_early_stopping_delta,
    ):
        init_key, seed = jr.split(seed)
        params = self._init_params(init_key, **next(iter(train_iter)))
        state = optimizer.init(params)

        loss_fn = jax.jit(
            partial(_cfm_loss, apply_fn=self.model.apply, is_training=True)
        )

        @jax.jit
        def step(params, rng, state, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, rng, **batch)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(
            n_early_stopping_delta, n_early_stopping_patience
        )
        best_params, best_loss = None, np.inf
        logging.info("training model")
        for i in tqdm(range(n_iter)):
            train_loss = 0.0
            rng_key = jr.fold_in(seed, i)
            for batch in train_iter:
                train_key, rng_key = jr.split(rng_key)
                batch_loss, params, state = step(
                    params, train_key, state, **batch
                )
                train_loss += batch_loss * (
                    batch["y"].shape[0] / train_iter.num_samples
                )
            val_key, rng_key = jr.split(rng_key)
            validation_loss = self._validation_loss(val_key, params, val_iter)
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

    def _init_params(self, rng_key, **init_data):
        times = jr.uniform(jr.PRNGKey(0), shape=(init_data["y"].shape[0], 1))
        params = self.model.init(
            rng_key,
            method="vector_field",
            theta=init_data["theta"],
            time=times,
            context=init_data["y"],
            is_training=False,
        )
        return params

    def _validation_loss(self, rng_key, params, val_iter):
        loss_fn = jax.jit(
            partial(_cfm_loss, apply_fn=self.model.apply, is_training=False)
        )

        def body_fn(batch_key, **batch):
            loss = loss_fn(params, batch_key, **batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        loss = 0.0
        for batch in val_iter:
            val_key, rng_key = jr.split(rng_key)
            loss += body_fn(val_key, **batch)
        return loss

    # ruff: noqa: D417
    def sample_posterior(
        self, rng_key, params, observable, *, n_samples=4_000, **kwargs
    ):
        r"""Sample from the approximate posterior.

        Args:
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: observation to condition on
            n_samples: number of samples to draw

        Returns:
            returns an array of samples from the posterior distribution of
            dimension (n_samples \times p)
        """
        observable = jnp.atleast_2d(observable)

        thetas = None
        n_curr = n_samples
        n_total_simulations_round = 0
        _, unravel_fn = ravel_pytree(self.prior_sampler_fn(seed=jr.PRNGKey(1)))
        while n_curr > 0:
            n_sim = jnp.minimum(200, jnp.maximum(200, n_curr))
            n_total_simulations_round += n_sim
            sample_key, rng_key = jr.split(rng_key)
            proposal = self.model.apply(
                params,
                sample_key,
                method="sample",
                context=jnp.tile(observable, [n_sim, 1]),
                is_training=False,
            )
            proposal_probs = self.prior_log_density_fn(
                jax.vmap(unravel_fn)(proposal)
            )
            proposal_accepted = proposal[jnp.isfinite(proposal_probs)]
            if thetas is None:
                thetas = proposal_accepted
            else:
                thetas = jnp.vstack([thetas, proposal_accepted])
            n_curr -= proposal_accepted.shape[0]
        self.n_total_simulations += n_total_simulations_round

        ess = float(thetas.shape[0] / n_total_simulations_round)
        thetas = jax.tree_map(
            lambda x: x.reshape(1, *x.shape),
            jax.vmap(unravel_fn)(thetas[:n_samples]),
        )
        inference_data = as_inference_data(thetas, jnp.squeeze(observable))
        return inference_data, ess
