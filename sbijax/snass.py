from functools import partial

import haiku as hk
import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr

from sbijax import SNL
from sbijax.generator import DataLoader
from sbijax.nn.early_stopping import EarlyStopping


def _cdist(x, y):
    return jax.vmap(
        jax.vmap(lambda a, b: jnp.linalg.norm(a - b, ord=2), in_axes=(None, 0)),
        in_axes=(0, None),
    )(x, y)


def _fn(val, m):
    mat = _cdist(val, val)
    row_sum = mat.sum(axis=0, keepdims=True)
    col_sum = mat.sum(axis=1, keepdims=True)
    ret = (
        mat
        - row_sum / (m - 2)
        - col_sum / (m - 2)
        + mat.sum() / ((m - 1) * (m - 2))
    )
    return ret


def _mean(val1, val2, m):
    return (val1 * val2).sum() / (m * (m - 3))


def _summary_loss(params, apply_fn, **batch):
    y, theta = batch["y"], batch["theta"]
    m, _ = y.shape
    summr = apply_fn(params, y)
    a = _fn(summr, m)
    b = _fn(theta, m)
    ab, aa, bb = _mean(a, b, m), _mean(a, a, m), _mean(b, b, m)
    mi = ab**0.5 / (aa**0.5 * bb**0.5) ** 0.5
    return -jnp.mean(mi)


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
    return mi


class SNASSNets:
    def __init__(self, dim):
        self.dim = dim
        self._summary = hk.nets.MLP(
            output_sizes=[64, 64, dim], activation=jax.nn.tanh
        )
        self._critic = hk.nets.MLP(
            output_sizes=[64, 64, 1], activation=jax.nn.tanh
        )

    def __call__(self, method, **kwargs):
        return getattr(self, method)(**kwargs)

    def forward(self, y, theta):
        s, c = self.summary(y), self.critic(y[:, : self.dim], theta)
        return s, c

    def summary(self, y):
        return self._summary(y)

    def critic(self, y, theta):
        return self._critic(jnp.concatenate([y, theta], axis=-1))


def make_critic(dim):
    @hk.without_apply_rng
    @hk.transform
    def _net(method, **kwargs):
        net = SNASSNets(dim)
        return net(method, **kwargs)

    return _net


# pylint: disable=too-many-arguments,unused-argument
class SNASS(SNL):
    """Sequential neural approximate summary statistics.

    References:
        .. [1] Yanzhi Chen et al. "Neural Approximate Sufficient Statistics for
           Implicit Models". ICLR, 2021
    """

    def __init__(self, model_fns, density_estimator, summary_net, critic_net):
        super().__init__(model_fns, density_estimator)
        # self.summary_net = summary_net
        # self.critic_net = critic_net
        self.sc_net = make_critic(2)
        self._s_params = {}

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
        """Fit a SNASS model.

        Args:
            rng_seq: a hk.PRNGSequence
            data: data set obtained from calling
              `simulate_data_and_possibly_append`
            optimizer: an optax optimizer object
            n_iter: maximal number of training iterations per round
            batch_size: batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated data
              that is used for valitation and early stopping
            n_early_stopping_patience: number of iterations of no improvement
              of training the flow before stopping optimisation
            kwargs: keyword arguments with sampler specific parameters. For
                sampling the following arguments are possible:
                - sampler: either 'nuts', 'slice' or None (defaults to nuts)
                - n_thin: number of thinning steps
                - n_doubling: number of doubling steps of the interval
                - step_size: step size of the initial interval

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

        train_iter = self._as_summary(train_iter, snet_params)
        val_iter = self._as_summary(val_iter, snet_params)

        nde_params, losses = self._fit_model_single_round(
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
        )

        self._s_params = snet_params
        return (nde_params, snet_params), (losses, snet_losses)

    def _as_summary(self, iter, params):
        @jax.jit
        def as_batch(y, theta):
            return {
                "y": self.sc_net.apply(params, method="summary", y=y),
                "theta": theta,
            }

        return DataLoader(
            num_batches=iter.num_batches,
            batches=[as_batch(**iter(i)) for i in range(iter.num_batches)],
        )

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
        params = self._init_summary_net_params(init_key, **train_iter(0))
        state = optimizer.init(params)
        loss_fn = jax.jit(
            partial(_jsd_summary_loss, apply_fn=self.sc_net.apply)
        )

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
            for j in range(train_iter.num_batches):
                batch = train_iter(j)
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

        losses = jnp.vstack(losses)[:i, :]
        return best_params, losses

    def _init_summary_net_params(self, rng_key, **init_data):
        params = self.sc_net.init(rng_key, method="forward", **init_data)
        return params

    def _summary_validation_loss(self, params, rng_key, val_iter):
        loss_fn = jax.jit(
            partial(_jsd_summary_loss, apply_fn=self.sc_net.apply)
        )

        def body_fn(i, batch_key):
            batch = val_iter(i)
            loss = loss_fn(params, batch_key, **batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        losses = 0.0
        for i in range(val_iter.num_batches):
            batch_key, rng_key = jr.split(rng_key)
            losses += body_fn(i, batch_key)
        return losses

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
        """Sample from the approximate posterior.

        Args:
            rng_key: a random key
            params: a pytree of parameter for the model
            observable: observation to condition on
            n_chains: number of MCMC chains
            n_samples: number of samples per chain
            n_warmup: number of samples to discard
            kwargs: keyword arguments with sampler specific parameters. For
                sampling the following arguments are possible:
                - sampler: either 'nuts', 'slice' or None (defaults to nuts)
                - n_thin: number of thinning steps
                - n_doubling: number of doubling steps of the interval
                - step_size: step size of the initial interval

        Returns:
            an array of samples from the posterior distribution of dimension
            (n_samples \times p)
        """

        observable = jnp.atleast_2d(observable)
        summary = self.sc_net.apply(
            self._s_params, method="summary", y=observable
        )
        return super().sample_posterior(
            rng_key,
            params,
            summary,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )
