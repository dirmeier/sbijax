from collections import namedtuple

import chex
import jax
from blackjax.smc import resampling
from blackjax.smc.ess import ess
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from jax import tree_map
from jax._src.flatten_util import ravel_pytree
from tensorflow_probability.substrates.jax import distributions as tfd
from tqdm import tqdm

from sbijax._src._sbi_base import SBI
from sbijax._src.util.data import _tree_stack, as_inference_data


# ruff: noqa: PLR0913, E501
class SMCABC(SBI):
    r"""Sequential Monte Carlo approximate Bayesian computation.

    Implements the algorithm from :cite:t:`beaumont2009adaptive`.

    Args:
        model_fns: a tuple of callables. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        summary_fn: summary function
        distance_fn: distance function

    Examples:
        >>> from sbijax import SMCABC
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...     dict(theta=tfd.Normal(0.0, 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> summary_fn = lambda x: x
        >>> distance_fn = lambda x, y: jax.vmap(lambda z: jnp.linalg.norm(z))(x - y)
        >>> model = SMCABC(fns, summary_fn, distance_fn)

    References:
        Beaumont, Mark A, et al. "Adaptive approximate Bayesian computation". Biometrika, 2009.
    """

    def __init__(self, model_fns, summary_fn, distance_fn):
        super().__init__(model_fns)
        self.summary_fn = summary_fn
        self.distance_fn = distance_fn
        self.summarized_observed: chex.Array
        self.n_total_simulations = 0

    def sample_posterior(
        self,
        rng_key,
        observable,
        n_rounds=10,
        n_particles=10_000,
        eps_step=0.825,
        ess_min=2_000,
        cov_scale=1.0,
    ):
        r"""Sample from the approximate posterior.

        Args:
            rng_key: a jax random
            n_rounds: max number of SMC rounds
            observable: the observation to condition on
            n_rounds: number of rounds of SMC
            n_particles: number of n_particles to draw for each parameter
            eps_step:  decay of initial epsilon per simulation round
            ess_min: minimal effective sample size
            cov_scale: scaling of the transition kernel covariance

        Returns:
            an array of samples from the posterior distribution of dimension
            (n_samples \times p)
        """
        observable = jnp.atleast_2d(observable)

        init_key, rng_key = jr.split(rng_key)
        particles, log_weights, epsilon = self._init_particles(
            init_key, observable, n_particles
        )

        all_particles, all_n_simulations = [], []
        for n in tqdm(range(n_rounds)):
            epsilon *= eps_step
            rng_key = jr.fold_in(rng_key, n)
            particle_key, rng_key = jr.split(rng_key)
            particles, log_weights = self._move(
                particle_key,
                observable,
                n_particles,
                particles,
                log_weights,
                epsilon,
                cov_scale,
            )
            curr_ess = ess(log_weights)
            if curr_ess < ess_min:
                resample_key, rng_key = jr.split(rng_key)
                particles[list(particles.keys())[0]]
                particles, log_weights = self._resample(
                    resample_key,
                    particles,
                    log_weights,
                    particles[list(particles.keys())[0]].shape[0],
                )
            all_particles.append(particles.copy())
            all_n_simulations.append(self.n_total_simulations)

        thetas = jax.tree_map(lambda x: x.reshape(1, *x.shape), particles)
        inference_data = as_inference_data(thetas, jnp.squeeze(observable))
        smc_info = namedtuple("smc_info", "particles n_simulations")
        return inference_data, smc_info(all_particles, all_n_simulations)

    def _chol_factor(self, particles, cov_scale):
        particles = jax.vmap(lambda x: ravel_pytree(x)[0])(particles)
        chol = jnp.linalg.cholesky(jnp.cov(particles.T) * cov_scale)
        return chol

    def _init_particles(self, rng_key, observable, n_particles):
        self.n_total_simulations += n_particles
        init_key, rng_key = jr.split(rng_key)
        particles = self.prior_sampler_fn(
            seed=init_key, sample_shape=(n_particles,)
        )
        simulator_key, rng_key = jr.split(rng_key)
        ys = self.simulator_fn(seed=simulator_key, theta=particles)

        summary_statistics = self.summary_fn(ys)
        distances = self.distance_fn(
            summary_statistics, self.summary_fn(observable)
        )

        sort_idx = jnp.argsort(distances)
        particles = jax.tree_map(lambda x: x[sort_idx][:n_particles], particles)
        log_weights = -jnp.log(jnp.full(n_particles, n_particles))
        initial_epsilon = distances[-1]

        return particles, log_weights, initial_epsilon

    def _sample_candidates(
        self, rng_key, particles, log_weights, n, cov_chol_factor
    ):
        n_sim = jnp.maximum(jnp.minimum(n, 1000), 100)
        self.n_total_simulations += n_sim

        sample_key, perturb_key, rng_key = jr.split(rng_key, 3)
        new_candidate_particles, _ = self._resample(
            sample_key, particles, log_weights, n_sim
        )
        new_candidate_particles = self._perturb(
            perturb_key, new_candidate_particles, cov_chol_factor
        )
        cand_lps = self.prior_log_density_fn(new_candidate_particles)
        is_finite = jnp.logical_not(jnp.isinf(cand_lps))
        new_candidate_particles = tree_map(
            lambda x: x[is_finite], new_candidate_particles
        )
        return new_candidate_particles

    def _simulate_and_distance(
        self, rng_key, observable, new_candidate_particles
    ):
        ys = self.simulator_fn(
            seed=rng_key,
            theta=new_candidate_particles,
        )
        summary_statistics = self.summary_fn(ys)
        ds = self.distance_fn(summary_statistics, self.summary_fn(observable))
        return ds

    # pylint: disable=too-many-arguments
    def _move(
        self,
        rng_key,
        observable,
        n_particles,
        particles,
        log_weights,
        epsilon,
        cov_scale,
    ):
        new_particles = None
        cov_chol_factor = self._chol_factor(particles, cov_scale)
        n = n_particles
        while n > 0:
            sample_key, simulate_key, rng_key = jr.split(rng_key, 3)
            new_candidate_particles = self._sample_candidates(
                sample_key, particles, log_weights, n, cov_chol_factor
            )
            ds = self._simulate_and_distance(
                simulate_key,
                observable,
                new_candidate_particles,
            )

            idxs = jnp.where(ds < epsilon)[0]
            new_candidate_particles = tree_map(
                lambda x: x[idxs], new_candidate_particles
            )
            if new_particles is None:
                new_particles = new_candidate_particles
            else:
                new_particles = _tree_stack(
                    [new_particles, new_candidate_particles]
                )
            n -= len(idxs)

        new_particles = tree_map(lambda x: x[:n_particles], new_particles)
        new_log_weights = self._new_log_weights(
            new_particles, particles, log_weights, cov_chol_factor
        )

        return new_particles, new_log_weights

    def _resample(self, rng_key, particles, log_weights, n_samples):
        idxs = resampling.multinomial(rng_key, jnp.exp(log_weights), n_samples)
        particles = tree_map(lambda x: x[idxs], particles)
        return particles, -jnp.log(jnp.full(n_samples, n_samples))

    def _new_log_weights(
        self, new_particles, old_particles, old_log_weights, cov_chol_factor
    ):
        prior_log_density = self.prior_log_density_fn(new_particles)
        K = self._kernel(old_particles, cov_chol_factor)

        def _particle_weight(partcl):
            probs = old_log_weights + K.log_prob(partcl)
            weight = jsp.special.logsumexp(probs)
            return weight

        new_particles = jax.vmap(lambda x: ravel_pytree(x)[0])(new_particles)
        new_particles = new_particles[:, None, :]
        log_weighted_sum = jax.vmap(_particle_weight)(new_particles)

        new_log_weights = prior_log_density - log_weighted_sum
        new_log_weights -= jsp.special.logsumexp(new_log_weights)
        return new_log_weights

    def _kernel(self, mus, cov_chol_factor):
        mus = jax.vmap(lambda x: ravel_pytree(x)[0])(mus)
        return tfd.MultivariateNormalTriL(loc=mus, scale_tril=cov_chol_factor)

    def _perturb(self, rng_key, mus, cov_chol_factor):
        _, unravel_fn = ravel_pytree(self.prior_sampler_fn(seed=jr.PRNGKey(0)))
        samples = self._kernel(mus, cov_chol_factor).sample(seed=rng_key)
        samples = jax.vmap(unravel_fn)(samples)
        return samples
