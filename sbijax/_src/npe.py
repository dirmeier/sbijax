from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from jax._src.flatten_util import ravel_pytree
from tqdm import tqdm

from sbijax._src._ne_base import NE
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.early_stopping import EarlyStopping


# ruff: noqa: PLR0913, E501
class NPE(NE):
    """Neural posterior estimation.

    Implements the method introduced in :cite:t:`greenberg2019automatic`.
    In the literature, the method is usually referred to as APT or NPE-C, but
    here we refer to it simply as NPE.

    Args:
        model_fns: a tuple of calalbles. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        density_estimator: a (neural) conditional density estimator
            to model the posterior distribution
        num_atoms: number of atomic atoms

    Examples:
        >>> from sbijax import NPE
        >>> from sbijax.nn import make_maf
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...     dict(theta=tfd.Normal(0.0, 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> neural_network = make_maf(1)
        >>> model = NPE(fns, neural_network)

    References:
        Greenberg, David, et al. "Automatic posterior transformation for likelihood-free inference." International Conference on Machine Learning, 2019.
    """

    def __init__(
        self,
        model_fns,
        density_estimator,
        num_atoms=10,
        use_event_space_bijections=True,
    ):
        """Construct an SNP object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            density_estimator: a (neural) conditional density estimator
                to model the posterior distribution
            num_atoms: number of atomic atoms
            use_event_space_bijections: if True uses a unconstraining bijection
                to map the constrained parameters onto the real line and do
                training there
        """
        super().__init__(model_fns, density_estimator)
        self.num_atoms = num_atoms
        self.n_round = 0
        prior = model_fns[0]()
        # TODO(simon): check out event bijections
        if (
            hasattr(prior, "experimental_default_event_space_bijector")
            and use_event_space_bijections
        ):
            self._prior_bijectors = (
                prior.experimental_default_event_space_bijector()
            )

    # ruff: noqa: D417
    def fit(
        self,
        rng_key,
        data,
        *,
        optimizer=optax.adam(0.0003),
        n_iter=1000,
        batch_size=128,
        percentage_data_as_validation_set=0.1,
        n_early_stopping_patience=10,
        **kwargs,
    ):
        """Fit an SNP model.

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
            seed=rng_key,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=optimizer,
            n_iter=n_iter,
            n_early_stopping_patience=n_early_stopping_patience,
            n_atoms=self.num_atoms,
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
        n_atoms,
    ):
        init_key, seed = jr.split(seed)
        params = self._init_params(init_key, **next(iter(train_iter)))
        state = optimizer.init(params)

        n_round = self.n_round
        _, unravel_fn = ravel_pytree(self.prior.sample(seed=jr.PRNGKey(1)))

        if n_round == 0:

            def loss_fn(params, rng, **batch):
                theta, y = batch["theta"], batch["y"]
                log_det = 0
                if hasattr(self, "_prior_bijectors"):
                    theta_map = jax.vmap(unravel_fn)(theta)
                    theta = self._prior_bijectors.inverse(theta_map)
                    log_det = self._prior_bijectors.inverse_log_det_jacobian(
                        theta_map
                    )
                    theta = jax.vmap(lambda x: ravel_pytree(x)[0])(theta)
                lp = self.model.apply(
                    params,
                    None,
                    method="log_prob",
                    y=theta,
                    x=y,
                )
                lp = lp + log_det
                return -jnp.mean(lp)

        else:
            # TODO(simon): do bijections here? probably
            def loss_fn(params, rng, **batch):
                lp = self._proposal_posterior_log_prob(
                    params,
                    rng,
                    n_atoms,
                    theta=batch["theta"],
                    y=batch["y"],
                )
                return -jnp.mean(lp)

        @jax.jit
        def step(params, rng, state, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, rng, **batch)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
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
            validation_loss = self._validation_loss(
                val_key, params, val_iter, n_atoms
            )
            losses[i] = jnp.array([train_loss, validation_loss])

            _, early_stop = early_stop.update(validation_loss)
            if early_stop.should_stop:
                logging.info("early stopping criterion found")
                break
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_params = params.copy()

        self.n_round += 1
        losses = jnp.vstack(losses)[: (i + 1), :]
        return best_params, losses

    def _init_params(self, rng_key, **init_data):
        params = self.model.init(
            rng_key, method="log_prob", y=init_data["theta"], x=init_data["y"]
        )
        return params

    def _proposal_posterior_log_prob(self, params, rng, n_atoms, theta, y):
        n = theta.shape[0]
        n_atoms = np.maximum(2, np.minimum(n_atoms, n))
        repeated_y = jnp.repeat(y, n_atoms, axis=0)
        probs = jnp.ones((n, n)) * (1 - jnp.eye(n)) / (n - 1)

        choice = partial(
            jr.choice, a=jnp.arange(n), replace=False, shape=(n_atoms - 1,)
        )
        sample_keys = jr.split(rng, probs.shape[0])
        choices = jax.vmap(lambda key, prob: choice(key, p=prob))(
            sample_keys, probs
        )
        contrasting_theta = theta[choices]

        atomic_theta = jnp.concatenate(
            (theta[:, None, :], contrasting_theta), axis=1
        )
        atomic_theta = atomic_theta.reshape(n * n_atoms, -1)

        log_prob_posterior = self.model.apply(
            params, None, method="log_prob", y=atomic_theta, x=repeated_y
        )
        log_prob_posterior = log_prob_posterior.reshape(n, n_atoms)
        log_prob_prior = self.prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(n, n_atoms)

        unnormalized_log_prob = log_prob_posterior - log_prob_prior
        log_prob_proposal_posterior = unnormalized_log_prob[
            :, 0
        ] - jsp.special.logsumexp(unnormalized_log_prob, axis=-1)

        return log_prob_proposal_posterior

    def _validation_loss(self, rng_key, params, val_iter, n_atoms):
        if self.n_round == 0:
            _, unravel_fn = ravel_pytree(self.prior.sample(seed=jr.PRNGKey(1)))

            def loss_fn(rng, **batch):
                theta, y = batch["theta"], batch["y"]
                log_det = 0
                if hasattr(self, "_prior_bijectors"):
                    theta_map = jax.vmap(unravel_fn)(theta)
                    theta = self._prior_bijectors.inverse(theta_map)
                    log_det = self._prior_bijectors.inverse_log_det_jacobian(
                        theta_map
                    )
                    theta = jax.vmap(lambda x: ravel_pytree(x)[0])(theta)
                lp = self.model.apply(
                    params,
                    None,
                    method="log_prob",
                    y=theta,
                    x=y,
                )
                lp = lp + log_det
                return -jnp.mean(lp)

        else:

            def loss_fn(rng, **batch):
                lp = self._proposal_posterior_log_prob(
                    params, rng, n_atoms, batch["theta"], batch["y"]
                )
                return -jnp.mean(lp)

        def body_fn(batch, rng_key):
            loss = jax.jit(loss_fn)(rng_key, **batch)
            return loss * (batch["y"].shape[0] / val_iter.num_samples)

        loss = 0.0
        for batch in val_iter:
            val_key, rng_key = jr.split(rng_key)
            loss += body_fn(batch, val_key)
        return loss

    def sample_posterior(
        self,
        rng_key,
        params,
        observable,
        *,
        n_samples=4_000,
        check_proposal_probs=True,
        **kwargs,
    ):
        r"""Sample from the approximate posterior.

        Args:
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: observation to condition on
            n_samples: number of samples to draw
            check_proposal_probs: check if the proposal draws have finite density
                and only accept a proposal if it is. This is convenient to turn
                off if the density estimator has to learn a density of a
                constrained variable, and the RejectionMC takes a long time to
                draw valid samples.

        Returns:
            returns an array of samples from the posterior distribution of
            dimension (n_samples \times p)
        """
        observable = jnp.atleast_2d(observable)

        thetas = None
        n_curr = n_samples
        n_total_simulations_round = 0
        _, unravel_fn = ravel_pytree(self.prior.sample(seed=jr.PRNGKey(1)))
        while n_curr > 0:
            n_sim = jnp.minimum(200, jnp.maximum(200, n_curr))
            n_total_simulations_round += n_sim
            sample_key, rng_key = jr.split(rng_key)
            proposal = self.model.apply(
                params,
                sample_key,
                method="sample",
                sample_shape=(n_sim,),
                x=jnp.tile(observable, [n_sim, 1]),
            )
            if hasattr(self, "_prior_bijectors"):
                proposal = jax.vmap(unravel_fn)(proposal)
                proposal = self._prior_bijectors.forward(proposal)
                proposal_probs = self.prior.log_prob(proposal)
                proposal = jax.vmap(lambda x: ravel_pytree(x)[0])(proposal)
            else:
                proposal_probs = self.prior_log_density_fn(
                    jax.vmap(unravel_fn)(proposal)
                )
            if check_proposal_probs:
                proposal = proposal[jnp.isfinite(proposal_probs)]
            if thetas is None:
                thetas = proposal
            else:
                thetas = jnp.vstack([thetas, proposal])
            n_curr -= proposal.shape[0]

        ess = float(thetas.shape[0] / n_total_simulations_round)

        def reshape(p):
            if p.ndim == 1:
                p = p.reshape(p.shape[0], 1)
            p = p.reshape(1, *p.shape)
            return p

        thetas = jax.tree_map(reshape, jax.vmap(unravel_fn)(thetas[:n_samples]))
        inference_data = as_inference_data(thetas, jnp.squeeze(observable))
        return inference_data, ess

    def _simulate_parameters_with_model(
        self,
        rng_key,
        params,
        observable,
        *,
        n_samples=4_000,
        check_proposal_probs=True,
        **kwargs,
    ):
        return self.sample_posterior(
            rng_key=rng_key,
            params=params,
            observable=observable,
            n_samples=n_samples,
            check_proposal_probs=check_proposal_probs,
            **kwargs,
        )
