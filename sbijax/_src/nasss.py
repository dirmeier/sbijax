from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.nass import NASS
from sbijax._src.util.early_stopping import EarlyStopping


def _sample_unit_sphere(rng_key, n, dim):
    u = jr.normal(rng_key, (n, dim))
    norm = jnp.linalg.norm(u, ord=2, axis=-1, keepdims=True)
    return u / norm


# pylint: disable=too-many-locals
def _jsd_summary_loss(params, rng_key, apply_fn, **batch):
    y, theta = batch["y"], batch["theta"]
    n, p = theta.shape

    phi_key, rng_key = jr.split(rng_key)
    summr = apply_fn(params, method="summary", y=y)
    summr = jnp.tile(summr, [10, 1])
    theta = jnp.tile(theta, [10, 1])

    phi = _sample_unit_sphere(phi_key, 10, p)
    phi = jnp.repeat(phi, n, axis=0)

    second_summr = apply_fn(
        params, method="secondary_summary", y=summr, theta=phi
    )
    theta_prime = jnp.sum(theta * phi, axis=1).reshape(-1, 1)

    idx_pos = jnp.tile(jnp.arange(n), 10)
    perm_key, rng_key = jr.split(rng_key)
    idx_neg = jax.vmap(lambda x: jr.permutation(x, n))(
        jr.split(perm_key, 10)
    ).reshape(-1)
    f_pos = apply_fn(params, method="critic", y=second_summr, theta=theta_prime)
    f_neg = apply_fn(
        params,
        method="critic",
        y=second_summr[idx_pos],
        theta=theta_prime[idx_neg],
    )
    a, b = -jax.nn.softplus(-f_pos), jax.nn.softplus(f_neg)
    mi = a.mean() - b.mean()
    return -mi


# ruff: noqa: PLR0913
class NASSS(NASS):
    """Neural approximate slice sufficient statistics.

    Args:
        model_fns: a tuple of tuples. The first element is a tuple that
                consists of functions to sample and evaluate the
                log-probability of a data point. The second element is a
                simulator function.
        density_estimator: a (neural) conditional density estimator
            to model the likelihood function of summary statistics, i.e.,
            the modelled dimensionality is that of the summaries
        summary_net: a SNASSSNet object

    References:
        .. [1] Yanzhi Chen et al. "Is Learning Summary Statistics Necessary for
           Likelihood-free Inference". ICML, 2023
    """

    # pylint: disable=useless-parent-delegation
    def __init__(self, model_fns, summary_net):
        """Construct a SNASSS object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            density_estimator: a (neural) conditional density estimator
                to model the likelihood function of summary statistics, i.e.,
                the modelled dimensionality is that of the summaries
            summary_net: a SNASSSNet object
        """
        super().__init__(model_fns, summary_net)

    # pylint: disable=undefined-loop-variable
    def _fit_summary_net(
        self,
        rng_key,
        train_iter,
        val_iter,
        optimizer,
        n_iter,
        n_early_stopping_patience,
    ):
        init_key, rng_key = jr.split(rng_key)
        params = self._init_summary_net_params(
            init_key, **next(iter(train_iter))
        )
        state = optimizer.init(params)
        loss_fn = jax.jit(partial(_jsd_summary_loss, apply_fn=self.model.apply))

        @jax.jit
        def step(rng, params, state, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, rng, **batch)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
        best_params, best_loss = None, np.inf
        logging.info("training summary net")
        for i in range(n_iter):
            train_loss = 0.0
            epoch_key, rng_key = jr.split(rng_key)
            for j, batch in enumerate(train_iter):
                batch_loss, params, state = step(
                    jr.fold_in(epoch_key, j), params, state, **batch
                )
                train_loss += batch_loss * (
                    batch["y"].shape[0] / train_iter.num_samples
                )
            val_key, rng_key = jr.split(rng_key)
            validation_loss = self._summary_validation_loss(
                params, val_key, val_iter
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

    def _summary_validation_loss(self, params, rng_key, val_iter):
        loss_fn = jax.jit(partial(_jsd_summary_loss, apply_fn=self.model.apply))

        def body_fn(batch_key, **batch):
            loss = loss_fn(params, batch_key, **batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        losses = 0.0
        for batch in val_iter:
            batch_key, rng_key = jr.split(rng_key)
            losses += body_fn(batch_key, **batch)
        return losses
