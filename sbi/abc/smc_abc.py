import chex
import distrax
import haiku as hk
import jax
from blackjax.smc import resampling
from blackjax.smc.ess import ess
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp


class SMCABC:
    """
    Sisson et al. - Handbook of approximate Bayesian computation

    Algorithm 4.8, "Algorithm 4.8: ABC Sequential Monte Carlo Algorithm"
    """

    def __init__(self, model_fns, summary_fn, distance_fn):
        self.prior_sampler_fn, self.prior_log_density_fn = model_fns[0]
        self.simulator_fn = model_fns[1]
        self.summary_fn = summary_fn
        self.distance_fn = distance_fn
        self._len_theta = len(self.prior_sampler_fn(seed=random.PRNGKey(0)))

        self.observed: chex.Array
        self.summarized_observed: chex.Array
        self._rng_seq: hk.PRNGSequence

    def fit(self, rng_key, observed):
        self._rng_seq = hk.PRNGSequence(rng_key)
        self.observed = jnp.atleast_2d(observed)
        self.summarized_observed = self.summary_fn(self.observed)

    def sample_posterior(
        self,
        n_total_simulations,
        n_samples,
        n_simulations_per_theta,
        eps_step,
        ess_min,
        cov_scale=1.0,
    ):
        particles, _, log_weights, epsilon = self.init_particles(
            n_samples, n_simulations_per_theta
        )
        all_particles = particles
        for _ in range(n_total_simulations):
            epsilon *= eps_step
            cov_chol_factor = jnp.linalg.cholesky(
                jnp.cov(particles.T) * cov_scale
            )
            particles, new_log_weights = self.move(
                n_samples,
                particles,
                log_weights,
                cov_chol_factor,
                n_simulations_per_theta,
                epsilon,
            )
            curr_ess = ess(log_weights)
            if curr_ess < ess_min:
                particles, log_weights = self.resample(
                    particles, log_weights, particles.shape[0]
                )
            all_particles = jnp.vstack([all_particles, particles])
        return all_particles[:n_samples, :]

    def init_particles(self, n_samples, n_simulations_per_theta):
        particles = self.prior_sampler_fn(
            seed=next(self._rng_seq), sample_shape=(n_samples * 10,)
        )
        ys = self.simulator_fn(
            seed=next(self._rng_seq),
            theta=jnp.tile(particles, [n_simulations_per_theta, 1, 1]),
        )
        ys = jnp.swapaxes(ys, 1, 0)
        summary_statistics = self.summary_fn(ys)
        distances = self.distance_fn(
            summary_statistics, self.summarized_observed
        )
        idxs = jnp.argsort(distances)

        particles = particles[idxs][:n_samples]
        summary_statistics = summary_statistics[idxs][:n_samples]
        log_weights = -jnp.log(jnp.full(n_samples, n_samples))
        initial_epsilon = distances[-1]

        return particles, summary_statistics, log_weights, initial_epsilon

    def move(
        self,
        n_samples,
        particles,
        log_weights,
        cov_chol_factor,
        n_simulations_per_theta,
        epsilon,
    ):
        new_particles = None
        n = n_samples
        while n > 0:
            n_sim = jnp.minimum(n, 1000)
            new_candidate_particles, _ = self.resample(
                particles, log_weights, n_sim
            )
            new_candidate_particles = self._perturb(
                new_candidate_particles, cov_chol_factor
            )
            cand_lps = self.prior_log_density_fn(new_candidate_particles)
            new_candidate_particles = new_candidate_particles[
                ~jnp.isinf(cand_lps)
            ]

            ys = self.simulator_fn(
                seed=next(self._rng_seq),
                theta=jnp.tile(
                    new_candidate_particles, [n_simulations_per_theta, 1, 1]
                ),
            )
            ys = jnp.swapaxes(ys, 1, 0)
            summary_statistics = self.summary_fn(ys)
            d = self.distance_fn(summary_statistics, self.summarized_observed)

            idxs = jnp.where(d < epsilon)[0]
            new_candidate_particles = new_candidate_particles[idxs]
            if new_particles is None:
                new_particles = new_candidate_particles
            else:
                new_particles = jnp.vstack(
                    [new_particles, new_candidate_particles]
                )
            n -= len(idxs)

        new_particles = new_particles[
            :n_samples,
        ]
        new_log_weights = self._new_log_weights(
            new_particles, particles, log_weights, cov_chol_factor
        )
        return new_particles, new_log_weights

    def resample(self, particles, log_weights, n_samples):
        idxs = resampling.multinomial(
            next(self._rng_seq), jnp.exp(log_weights), n_samples
        )
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

    def _perturb(self, mus, cov_chol_factor):
        return self._kernel(mus, cov_chol_factor).sample(
            seed=next(self._rng_seq)
        )
