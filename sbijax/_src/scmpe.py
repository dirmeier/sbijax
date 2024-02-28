from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from sbijax._src._sne_base import SNE
from sbijax._src.util.early_stopping import EarlyStopping


def _alpha_t(time):
    return 1 / (_time_schedule(time + 1) - _time_schedule(time))


def _time_schedule(n, rho=7, eps=0.001, T_max=50, N=1000):
    left = eps ** (1 / rho)
    right = T_max ** (1 / rho) - eps ** (1 / rho)
    right = (n - 1) / (N - 1) * right
    return (left + right) ** rho


# pylint: disable=too-many-locals
def _consistency_loss(
    params, ema_params, rng_key, apply_fn, is_training=False, **batch
):
    theta = batch["theta"]

    t_key, rng_key = jr.split(rng_key)
    time_idx = jr.randint(t_key, shape=(theta.shape[0],), minval=1, maxval=1000 - 1)
    tn = _time_schedule(time_idx).reshape(-1, 1)
    tnp1 = _time_schedule(time_idx + 1).reshape(-1, 1)

    noise_key, rng_key = jr.split(rng_key)
    noise = jr.normal(noise_key, shape=(*theta.shape,))

    train_rng, rng_key = jr.split(rng_key)
    fnp1 = apply_fn(
        params,
        train_rng,
        method="vector_field",
        theta=theta + tnp1 * noise,
        time=tnp1,
        context=batch["y"],
        is_training=is_training,
    )
    fn = apply_fn(
        ema_params,
        train_rng,
        method="vector_field",
        theta=theta + tn * noise,
        time=tn,
        context=batch["y"],
        is_training=is_training,
    )
    mse = jnp.mean(jnp.square(fnp1 - fn), axis=1)
    loss = _alpha_t(time_idx) * mse
    return jnp.mean(loss)


# pylint: disable=too-many-arguments,unused-argument,useless-parent-delegation
class SCMPE(SNE):
    r"""Sequential consistency model posterior estimation.

    Implements a sequential version of the CMPE algorithm introduced in [1]_.
    For all rounds $r > 1$ parameter samples
    :math:`\theta \sim \hat{p}^r(\theta)` are drawn from
    the approximate posterior instead of the prior when computing consistency
    loss

    Args:
        model_fns: a tuple of tuples. The first element is a tuple that
                consists of functions to sample and evaluate the
                log-probability of a data point. The second element is a
                simulator function.
        network: a neural network

    Examples:
        >>> import distrax
        >>> from sbijax import SCMPE
        >>> from sbijax.nn import make_consistency_model
        >>>
        >>> prior = distrax.Normal(0.0, 1.0)
        >>> s = lambda seed, theta: distrax.Normal(theta, 1.0).sample(seed=seed)
        >>> fns = (prior.sample, prior.log_prob), s
        >>> net = make_consistency_model(1)
        >>>
        >>> estim = SCMPE(fns, net)

    References:
        .. [1] Wildberger, Jonas, et al. "Flow Matching for Scalable
           Simulation-Based Inference." Advances in Neural Information
           Processing Systems, 2024.
    """

    def __init__(self, model_fns, network):
        """Construct a FMPE object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            network: network: a neural network
        """
        super().__init__(model_fns, network)

    # pylint: disable=arguments-differ,too-many-locals
    def fit(
        self,
        rng_key,
        data,
        *,
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
            batch_size:  batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated
                data that is used for validation and early stopping
            n_early_stopping_patience: number of iterations of no improvement
                of training the flow before stopping optimisation\

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
        )

        return params, losses

    # pylint: disable=undefined-loop-variable
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
        ema_params = params.copy()
        state = optimizer.init(params)

        loss_fn = jax.jit(
            partial(
                _consistency_loss, apply_fn=self.model.apply, is_training=False
            )
        )

        @jax.jit
        def ema_update(params, avg_params):
            return optax.incremental_update(params, avg_params, step_size=0.001)

        @jax.jit
        def step(params, ema_params, rng, state, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(
                params, ema_params, rng, **batch
            )
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            new_ema_params = ema_update(new_params, ema_params)
            return loss, new_params, new_ema_params, new_state

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
        best_params, best_loss = None, np.inf
        logging.info("training model")
        for i in tqdm(range(n_iter)):
            train_loss = 0.0
            rng_key = jr.fold_in(seed, i)
            for batch in train_iter:
                train_key, rng_key = jr.split(rng_key)
                batch_loss, params, ema_params, state = step(
                    params, ema_params, train_key, state, **batch
                )
                train_loss += batch_loss * (
                    batch["y"].shape[0] / train_iter.num_samples
                )
            val_key, rng_key = jr.split(rng_key)
            validation_loss = self._validation_loss(
                val_key, params, ema_params, val_iter
            )
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

    def _validation_loss(self, rng_key, params, ema_params, val_iter):
        loss_fn = jax.jit(
            partial(
                _consistency_loss, apply_fn=self.model.apply, is_training=False
            )
        )

        def body_fn(batch_key, **batch):
            loss = loss_fn(params, ema_params, batch_key, **batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        loss = 0.0
        for batch in val_iter:
            val_key, rng_key = jr.split(rng_key)
            loss += body_fn(val_key, **batch)
        return loss

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
        while n_curr > 0:
            n_sim = jnp.minimum(200, jnp.maximum(200, n_curr))
            n_total_simulations_round += n_sim
            sample_key, rng_key = jr.split(rng_key)
            proposal = self.model.apply(
                params,
                sample_key,
                method="sample",
                context=jnp.tile(observable, [n_sim, 1]),
            )
            proposal_probs = self.prior_log_density_fn(proposal)
            proposal_accepted = proposal[jnp.isfinite(proposal_probs)]
            if thetas is None:
                thetas = proposal_accepted
            else:
                thetas = jnp.vstack([thetas, proposal_accepted])
            n_curr -= proposal_accepted.shape[0]

        self.n_total_simulations += n_total_simulations_round
        return (
            thetas[:n_samples],
            thetas.shape[0] / n_total_simulations_round,
        )
