from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from sbijax._src.fmpe import FMPE
from sbijax._src.util.early_stopping import EarlyStopping


def _alpha_t(time):
    return 1.0 / (_time_schedule(time + 1) - _time_schedule(time))


def _time_schedule(n, rho=7, t_min=0.001, t_max=50, n_inters=1000):
    left = t_min ** (1 / rho)
    right = t_max ** (1 / rho) - t_min ** (1 / rho)
    right = (n - 1) / (n_inters - 1) * right
    return (left + right) ** rho


def _discretization_schedule(n_iter, max_iter=1000):
    s0, s1 = 10, 50
    nk = (
        (n_iter / max_iter) * (jnp.square(s1 + 1) - jnp.square(s0))
        + jnp.square(s0)
        - 1
    )
    nk = jnp.ceil(jnp.sqrt(nk)) + 1
    return nk


# ruff: noqa: PLR0913
def _consistency_loss(
    params,
    ema_params,
    rng_key,
    apply_fn,
    n_iter,
    t_min,
    t_max,
    is_training=False,
    **batch,
):
    theta = batch["theta"]
    nk = _discretization_schedule(n_iter)

    t_key, rng_key = jr.split(rng_key)
    time_idx = jr.randint(
        t_key, shape=(theta.shape[0],), minval=1, maxval=nk - 1
    )
    tn = _time_schedule(
        time_idx, t_min=t_min, t_max=t_max, n_inters=nk
    ).reshape(-1, 1)
    tnp1 = _time_schedule(
        time_idx + 1, t_min=t_min, t_max=t_max, n_inters=nk
    ).reshape(-1, 1)

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
    mse = jnp.sqrt(jnp.mean(jnp.square(fnp1 - fn), axis=1))
    loss = _alpha_t(time_idx) * mse
    return jnp.mean(loss)


# ruff: noqa: E501
class CMPE(FMPE):
    r"""Consistency model posterior estimation.

    Implements the CMPE algorithm introduced in
    :cite:t:`schmitt2023con`.

    Args:
        model_fns: a tuple of callables. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        network: a consistency model
        t_min: minimal time point for ODE integration
        t_max: maximal time point for ODE integration

    Examples:
        >>> from sbijax import CMPE
        >>> from sbijax.nn import make_cm
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...     dict(theta=tfd.Normal(0.0, 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> neural_network = make_cm(1)
        >>> model = CMPE(fns, neural_network)

    References:
        Schmitt, Marvin, et al. "Consistency Models for Scalable and Fast Simulation-Based Inference". arXiv preprint arXiv:2312.05440, 2023.
    """

    def __init__(self, model_fns, network, t_max=50.0, t_min=0.001):
        super().__init__(model_fns, network)
        self._t_min = t_min
        self._t_max = t_max

    # ruff: noqa: PLR0913
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
        ema_params = params.copy()
        state = optimizer.init(params)

        loss_fn = jax.jit(
            partial(
                _consistency_loss,
                apply_fn=self.model.apply,
                is_training=True,
                t_max=self._t_max,
                t_min=self._t_min,
            )
        )

        @jax.jit
        def ema_update(params, avg_params):
            return optax.incremental_update(avg_params, params, step_size=0.01)

        @jax.jit
        def step(params, ema_params, rng, state, n_iter, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(
                params, ema_params, rng, n_iter=n_iter, **batch
            )
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            new_ema_params = ema_update(new_params, ema_params)
            return loss, new_params, new_ema_params, new_state

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
                batch_loss, params, ema_params, state = step(
                    params, ema_params, train_key, state, n_iter + 1, **batch
                )
                train_loss += batch_loss * (
                    batch["y"].shape[0] / train_iter.num_samples
                )
            val_key, rng_key = jr.split(rng_key)
            validation_loss = self._validation_loss(
                val_key, params, ema_params, n_iter, val_iter
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
            is_training=True,
        )
        return params

    # ruff: noqa: PLR0913
    def _validation_loss(self, rng_key, params, ema_params, n_iter, val_iter):
        loss_fn = jax.jit(
            partial(
                _consistency_loss,
                apply_fn=self.model.apply,
                is_training=False,
                t_max=self._t_max,
                t_min=self._t_min,
                n_iter=n_iter,
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
