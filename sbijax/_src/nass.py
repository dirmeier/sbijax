from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from sbijax._src._ne_base import NE
from sbijax._src.util.dataloader import as_numpy_iterator_from_slices
from sbijax._src.util.early_stopping import EarlyStopping


def _jsd_summary_loss(params, rng, apply_fn, **batch):
    y, theta = batch["y"], batch["theta"]
    m, _ = y.shape
    summr = apply_fn(params, method="summary", y=y)
    idx_pos = jnp.tile(jnp.arange(m), 10)
    idx_neg = jax.vmap(lambda x: jr.permutation(x, m))(
        jr.split(rng, 10)
    ).reshape(-1)
    f_pos = apply_fn(params, method="critic", y=summr, theta=theta)
    f_neg = apply_fn(
        params, method="critic", y=summr[idx_pos], theta=theta[idx_neg]
    )
    a, b = -jax.nn.softplus(-f_pos), jax.nn.softplus(f_neg)
    mi = a.mean() - b.mean()
    return -mi


# ruff: noqa: PLR0913, E501
class NASS(NE):
    """Neural approximate summary statistics.

    Implements the NASS algorithm introduced in :cite:t:`chen2023learning`.
    NASS can be used to automatically summary statistics of a data set.
    With the learned summaries, inferential algorithms like NLE or SMCABC
    can be used to infer posterior distributions.

    Args:
        model_fns: a tuple of calalbles. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        summary_net: a SNASSNet object

    Examples:
        >>> from sbijax import NASS
        >>> from sbijax.nn import make_nass_net
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...    dict(theta=tfd.Normal(jnp.zeros(5), 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(
        ...     theta["theta"], 1.0).sample(seed=seed, sample_shape=(2,)
        ... ).reshape(-1, 10)
        >>> fns = prior, s
        >>> neural_network = make_nass_net([64, 64, 5], [64, 64, 1])
        >>> model = NASS(fns, neural_network)

    References:
        Chen, Yanzhi et al. "Neural Approximate Sufficient Statistics for Implicit Models". ICLR, 2021
    """

    def __init__(self, model_fns, summary_net):
        super().__init__(model_fns, summary_net)

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
        """Fit the model to data.

        Args:
            rng_key: a jax random key
            data: data set obtained from calling
              `simulate_data_and_possibly_append`
            optimizer: an optax optimizer object
            n_iter: maximal number of training iterations per round
            batch_size: batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated data
              that is used for validation and early stopping
            n_early_stopping_patience: number of iterations of no improvement
              of training the flow before stopping optimisation
            **kwargs: additional keyword arguments not used for NASS)

        Returns:
            tuple of parameters and a tuple of the training information
        """
        itr_key, rng_key = jr.split(rng_key)
        train_iter, val_iter = self.as_iterators(
            itr_key, data, batch_size, percentage_data_as_validation_set
        )

        snet_params, snet_losses = self._fit_summary_net(
            rng_key=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
        )
        return snet_params, snet_losses

    # TODO(Simon): this is not very nicely solved
    def summarize(self, params, data, batch_size=512):
        if params is None or len(params) == 0:
            return data
        y = {"y": data} if isinstance(data, jnp.ndarray) else data
        itr = as_numpy_iterator_from_slices(y, batch_size)

        @jax.jit
        def _summarize(batch):
            return self.model.apply(params, method="summary", y=batch["y"])

        summaries = jnp.concatenate(
            [_summarize(batch) for batch in itr], axis=0
        )

        if isinstance(data, dict):
            ret_summaries = data.copy()
            ret_summaries["y"] = summaries
        else:
            ret_summaries = summaries

        return ret_summaries

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
        for i in tqdm(range(n_iter)):
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

    def _init_summary_net_params(self, rng_key, **init_data):
        params = self.model.init(rng_key, method="forward", **init_data)
        return params

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

    def simulate_data(
        self,
        rng_key,
        *,
        n_simulations=1000,
        **kwargs,
    ):
        return super().simulate_data(
            rng_key, n_simulations=n_simulations, **kwargs
        )

    def _simulate_parameters_with_model(
        self, rng_key, params, observable, *args, **kwargs
    ):
        raise NotImplementedError()

    def sample_posterior(self, rng_key, params, observable, *args, **kwargs):
        raise NotImplementedError()
