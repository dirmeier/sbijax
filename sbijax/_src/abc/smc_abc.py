from collections import namedtuple

import chex
import distrax
import jax
from blackjax.smc import resampling
from blackjax.smc.ess import ess
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp

from sbijax._src._sbi_base import SBI


# pylint: disable=arguments-differ,too-many-function-args,too-many-locals
# pylint: disable=too-few-public-methods
class SMCABC(SBI):
    """Sequential Monte Carlo approximate Bayesian computation.

    Implements algorithm~4.8 from [1].

    References:
        .. [1] Sisson, Scott A, et al. "Handbook of approximate Bayesian
           computation". 2019
    """

    def __init__(self, model_fns, summary_fn, distance_fn):
        """Construct a SMCABC object.

        Args:
            model_fns: tuple
            summary_fn: summary function
            distance_fn: distance function
        """
        super().__init__(model_fns)
        self.summary_fn = summary_fn
        self.distance_fn = distance_fn
        self.summarized_observed: chex.Array
        self.n_total_simulations = 0

    # pylint: disable=too-many-arguments,arguments-differ
    def sample_posterior(
        self,
        rng_key,
        observable,
        n_rounds,
        n_particles,
        n_simulations_per_theta,
        eps_step,
        ess_min,
        cov_scale=1.0,
    ):
        r"""Sample from the approximate posterior.

        Args:
            n_rounds: max number of SMC rounds
            observable: the observation to condition on
            n_round: number of rounds of SMC
            n_particles: number of n_particles to draw for each parameter
            n_simulations_per_theta: number of simulations for each paramrter
                sample
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
            init_key, observable, n_particles, n_simulations_per_theta
        )

        all_particles, all_n_simulations = [], []
        for n in range(n_rounds):
            epsilon *= eps_step
            rng_key = jr.fold_in(rng_key, n)
            particle_key, rng_key = jr.split(rng_key)
            particles, log_weights = self._move(
                particle_key,
                observable,
                n_particles,
                particles,
                log_weights,
                n_simulations_per_theta,
                epsilon,
                cov_scale,
            )
            curr_ess = ess(log_weights)
            if curr_ess < ess_min:
                resample_key, rng_key = jr.split(rng_key)
                particles, log_weights = self._resample(
                    resample_key, particles, log_weights, particles.shape[0]
                )
            all_particles.append(particles.copy())
            all_n_simulations.append(self.n_total_simulations)

        smc_info = namedtuple("smc_info", "particles n_simulations")
        return particles, smc_info(all_particles, all_n_simulations)

    def _chol_factor(self, particles, cov_scale):
        chol = jnp.linalg.cholesky(jnp.cov(particles.T) * cov_scale)
        return chol

    def _init_particles(
        self, rng_key, observable, n_particles, n_simulations_per_theta
    ):
        self.n_total_simulations += n_particles * 10

        init_key, rng_key = jr.split(rng_key)
        particles = self.prior_sampler_fn(
            seed=init_key, sample_shape=(n_particles * 10,)
        )

        thetas = jnp.tile(particles, [n_simulations_per_theta, 1, 1])
        chex.assert_axis_dimension(thetas, 0, n_simulations_per_theta)
        chex.assert_axis_dimension(thetas, 1, n_particles * 10)

        simulator_key, rng_key = jr.split(rng_key)
        ys = self.simulator_fn(seed=simulator_key, theta=thetas)
        ys = jnp.swapaxes(ys, 1, 0)
        chex.assert_axis_dimension(ys, 0, n_particles * 10)
        chex.assert_axis_dimension(ys, 1, n_simulations_per_theta)

        summary_statistics = self.summary_fn(ys)
        distances = self.distance_fn(
            summary_statistics, self.summary_fn(observable)
        )

        sort_idx = jnp.argsort(distances)
        particles = particles[sort_idx][:n_particles]
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
        new_candidate_particles = new_candidate_particles[
            jnp.logical_not(jnp.isinf(cand_lps))
        ]
        return new_candidate_particles

    def _simulate_and_distance(
        self,
        rng_key,
        observable,
        new_candidate_particles,
        n_simulations_per_theta,
    ):
        ys = self.simulator_fn(
            seed=rng_key,
            theta=jnp.tile(
                new_candidate_particles, [n_simulations_per_theta, 1, 1]
            ),
        )
        ys = jnp.swapaxes(ys, 1, 0)
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
        n_simulations_per_theta,
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
                n_simulations_per_theta,
            )

            idxs = jnp.where(ds < epsilon)[0]
            new_candidate_particles = new_candidate_particles[idxs]
            if new_particles is None:
                new_particles = new_candidate_particles
            else:
                new_particles = jnp.vstack(
                    [new_particles, new_candidate_particles]
                )
            n -= len(idxs)

        new_particles = new_particles[:n_particles,]
        new_log_weights = self._new_log_weights(
            new_particles, particles, log_weights, cov_chol_factor
        )

        return new_particles, new_log_weights

    def _resample(self, rng_key, particles, log_weights, n_samples):
        idxs = resampling.multinomial(rng_key, jnp.exp(log_weights), n_samples)
        particles = particles[idxs]
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

        new_particles = new_particles[:, None, :]
        log_weighted_sum = jax.vmap(_particle_weight)(new_particles)

        new_log_weights = prior_log_density - log_weighted_sum
        new_log_weights -= jsp.special.logsumexp(new_log_weights)
        return new_log_weights

    def _kernel(self, mus, cov_chol_factor):
        return distrax.MultivariateNormalTri(loc=mus, scale_tri=cov_chol_factor)

    def _perturb(self, rng_key, mus, cov_chol_factor):
        return self._kernel(mus, cov_chol_factor).sample(seed=rng_key)
