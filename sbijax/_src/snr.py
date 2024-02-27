# Parts of this codebase have been adopted from https://github.com/bkmi/cnre
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import chex
import jax
import numpy as np
import optax
from absl import logging
from haiku import Params, Transformed
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from tqdm import tqdm

from sbijax._src import mcmc
from sbijax._src._sne_base import SNE
from sbijax._src.mcmc import mcmc_diagnostics
from sbijax._src.util.early_stopping import EarlyStopping


def _get_prior_probs_marginal_and_joint(K, gamma):
    p_marginal = 1 / (1 + gamma * K)
    p_joint = gamma / (1 + gamma * K)
    return p_marginal, p_joint


# pylint: disable=too-many-arguments
def _as_logits(params, rng_key, model, K, theta, y):
    n = theta.shape[0]
    y = jnp.repeat(y, K + 1, axis=0)
    ps = jnp.ones((n, n)) * (1.0 - jnp.eye(n)) / (n - 1.0)

    choices = jax.vmap(
        lambda key, p: jr.choice(key, n, (K,), replace=False, p=p)
    )(jr.split(rng_key, n), ps)

    contrasting_theta = theta[choices]
    atomic_theta = jnp.concatenate(
        [theta[:, None, :], contrasting_theta], axis=1
    ).reshape(n * (K + 1), -1)

    inputs = jnp.concatenate([y, atomic_theta], axis=-1)
    return model.apply(params, inputs, is_training=False)


def _marginal_joint_loss(gamma, num_classes, log_marg, log_joint):
    loggamma = jnp.log(gamma)
    logK = jnp.full((log_marg.shape[0], 1), jnp.log(num_classes))

    denominator_marginal = jnp.concatenate(
        [loggamma + log_marg, logK],
        axis=-1,
    )
    denomintator_joint = jnp.concatenate(
        [loggamma + log_joint, logK],
        axis=-1,
    )

    log_prob_marginal = logK - jsp.special.logsumexp(
        denominator_marginal, axis=-1
    )
    log_prob_joint = (
        loggamma
        + log_joint[:, 0]
        - jsp.special.logsumexp(denomintator_joint, axis=-1)
    )

    p_marg, p_joint = _get_prior_probs_marginal_and_joint(num_classes, gamma)
    loss = p_marg * log_prob_marginal + p_joint * num_classes * log_prob_joint
    return loss


def _loss(params, rng_key, model, gamma, num_classes, **batch):
    n, _ = batch["y"].shape

    rng_key1, rng_key2, rng_key = jr.split(rng_key, 3)
    log_marg = _as_logits(params, rng_key1, model, num_classes, **batch)
    log_joint = _as_logits(params, rng_key2, model, num_classes, **batch)

    log_marg = log_marg.reshape(n, num_classes + 1)[:, 1:]
    log_joint = log_joint.reshape(n, num_classes + 1)[:, :-1]

    loss = _marginal_joint_loss(gamma, num_classes, log_marg, log_joint)
    return -jnp.mean(loss)


# pylint: disable=too-many-arguments,unused-argument
class SNR(SNE):
    r"""Sequential (contrastive) neural ratio estimation.

    Args:
        model_fns: a tuple of tuples. The first element is a tuple that
                consists of functions to sample and evaluate the
                log-probability of a data point. The second element is a
                simulator function.
        classifier: a neural network for classification
        num_classes: number of classes to classify against
        gamma: relative weight of classes

    Examples:
        >>> import distrax
        >>> from sbijax import SNR
        >>> from sbijax.nn import make_resnet
        >>>
        >>> prior = distrax.Normal(0.0, 1.0)
        >>> s = lambda seed, theta: distrax.Normal(theta, 1.0).sample(seed=seed)
        >>> fns = (prior.sample, prior.log_prob), s
        >>> resnet = make_resnet()
        >>>
        >>> snr = SNR(fns, resnet)

    References:
        .. [1] Miller, Benjamin K., et al. "Contrastive neural ratio
           estimation." Advances in Neural Information Processing Systems, 2022.
    """

    def __init__(
        self,
        model_fns: Tuple[Tuple[Callable, Callable], Callable],
        classifier: Transformed,
        num_classes: int = 10,
        gamma: float = 1.0,
    ):
        """Construct an SNR object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                consists of functions to sample and evaluate the
                log-probability of a data point. The second element is a
                simulator function.
            classifier: a neural network for classification
            num_classes: number of classes to classify against
            gamma: relative weight of classes
        """
        super().__init__(model_fns, classifier)
        self.gamma = gamma
        self.num_classes = num_classes

    # pylint: disable=arguments-differ,too-many-locals
    def fit(
        self,
        rng_key: Array,
        data: NamedTuple,
        *,
        optimizer: optax.GradientTransformation = optax.adam(0.003),
        n_iter: int = 1000,
        batch_size: int = 100,
        percentage_data_as_validation_set: float = 0.1,
        n_early_stopping_patience: float = 10,
        **kwargs,
    ):
        """Fit an SNR model.

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

        Returns:
            a tuple of parameters and a tuple of the training information
        """
        itr_key, rng_key = jr.split(rng_key)
        train_iter, val_iter = self.as_iterators(
            itr_key, data, batch_size, percentage_data_as_validation_set
        )
        params, losses = self._fit_model_single_round(
            rng_key=rng_key,
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
        rng_key,
        train_iter,
        val_iter,
        optimizer,
        n_iter,
        n_early_stopping_patience,
    ):
        init_key, rng_key = jr.split(rng_key)
        params = self._init_params(init_key, **next(iter(train_iter)))
        state = optimizer.init(params)

        loss_fn = partial(_loss, gamma=self.gamma, num_classes=self.num_classes)

        @jax.jit
        def step(params, rng, state, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(
                params, rng, self.model, **batch
            )
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
        best_params, best_loss = None, np.inf
        logging.info("training model")
        for i in tqdm(range(n_iter)):
            train_loss = 0.0
            rng_key = jr.fold_in(rng_key, i)
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
        params = self.model.init(
            rng_key,
            jnp.concatenate([init_data["y"], init_data["theta"]], axis=-1),
        )
        return params

    def _validation_loss(self, rng_key, params, val_iter):
        loss_fn = partial(_loss, gamma=self.gamma, num_classes=self.num_classes)

        @jax.jit
        def body_fn(rng_key, **batch):
            loss = loss_fn(params, rng_key, self.model, **batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        loss = 0.0
        for batch in val_iter:
            val_key, rng_key = jr.split(rng_key)
            loss += body_fn(val_key, **batch)
        return loss

    def simulate_data_and_possibly_append(
        self,
        rng_key: Array,
        params: Optional[Params] = None,
        observable: Array = None,
        data: NamedTuple = None,
        n_simulations: int = 1_000,
        n_chains: int = 4,
        n_samples: int = 2_000,
        n_warmup: int = 1_000,
        **kwargs,
    ):
        """Simulate data from the prior or posterior.

        Simulate new parameters and observables from the prior or posterior
        (when params and data given). If a data argument is provided, append
        the new samples to the data set and return the old+new data.

        Args:
            rng_key: a jax random key
            params: a dictionary of neural network parameters
            observable: an observation
            data: existing data set or None
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
            returns an array of samples from the posterior distribution of
            dimension (n_samples \times p) and posterior diagnostics
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
        part = partial(self.model.apply, params, is_training=False)

        def _joint_logdensity_fn(theta):
            lp_prior = self.prior_log_density_fn(theta)
            theta = theta.reshape(observable.shape)
            lp = part(jnp.concatenate([observable, theta], axis=-1))
            return jnp.sum(lp_prior) + jnp.sum(lp)

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
