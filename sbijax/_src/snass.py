from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from sbijax._src.snl import SNL
from sbijax._src.util.dataloader import as_batch_iterator, named_dataset
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


# pylint: disable=too-many-arguments,unused-argument
class SNASS(SNL):
    """Sequential neural approximate summary statistics.

    Args:
        model_fns: a tuple of tuples. The first element is a tuple that
                consists of functions to sample and evaluate the
                log-probability of a data point. The second element is a
                simulator function.
        density_estimator: a (neural) conditional density estimator
            to model the likelihood function of summary statistics, i.e.,
            the modelled dimensionality is that of the summaries
        snass_net: a SNASSNet object

    References:
        .. [1] Chen, Yanzhi et al. "Neural Approximate Sufficient Statistics for
           Implicit Models". ICLR, 2021
    """

    def __init__(self, model_fns, density_estimator, summary_net):
        """Construct a SNASS object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            density_estimator: a (neural) conditional density estimator
                to model the likelihood function of summary statistics, i.e.,
                the modelled dimensionality is that of the summaries
            summary_net: a SNASSNet object
        """
        super().__init__(model_fns, density_estimator)
        self.sc_net = summary_net

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

        Keyword Args:
            sampler (str): either 'nuts', 'slice' or None (defaults to nuts)
            n_thin (int): number of thinning steps
                (only used if sampler='slice')
            n_doubling (int): number of doubling steps of the interval
                 (only used if sampler='slice')
            step_size (float): step size of the initial interval
                 (only used if sampler='slice')

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

        train_key, val_key, rng_key = jr.split(rng_key, 3)
        train_iter = self._as_itr_over_summaries(
            train_key, train_iter, snet_params, batch_size
        )
        val_iter = self._as_itr_over_summaries(
            val_key, val_iter, snet_params, batch_size
        )

        nde_params, losses = self._fit_model_single_round(
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
        )

        return {"params": nde_params, "s_params": snet_params}, (
            losses,
            snet_losses,
        )

    def _as_itr_over_summaries(self, rng_key, iters, params, batch_size):
        ys = []
        thetas = []
        for batch in iters:
            ys.append(self.sc_net.apply(params, method="summary", y=batch["y"]))
            thetas.append(batch["theta"])
        ys = jnp.vstack(ys)
        thetas = jnp.vstack(thetas)
        return as_batch_iterator(
            rng_key, named_dataset(ys, thetas), batch_size, True
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
        params = self._init_summary_net_params(
            init_key, **next(iter(train_iter))
        )
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
        params = self.sc_net.init(rng_key, method="forward", **init_data)
        return params

    def _summary_validation_loss(self, params, rng_key, val_iter):
        loss_fn = jax.jit(
            partial(_jsd_summary_loss, apply_fn=self.sc_net.apply)
        )

        def body_fn(batch_key, **batch):
            loss = loss_fn(params, batch_key, **batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        losses = 0.0
        for batch in val_iter:
            batch_key, rng_key = jr.split(rng_key)
            losses += body_fn(batch_key, **batch)
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
        r"""Sample from the approximate posterior.

        Args:
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: observation to condition on
            n_chains: number of MCMC chains
            n_samples: number of samples per chain
            n_warmup: number of samples to discard

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
        summary = self.sc_net.apply(
            params["s_params"], method="summary", y=observable
        )
        return super().sample_posterior(
            rng_key,
            params["params"],
            summary,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )
