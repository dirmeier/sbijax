# pylint: skip-file
import chex
from jax import random as jr

from sbijax._src.mcmc import sample_with_imh


def test_rmh_sampler(prior_log_prob_tuple):
    samples = sample_with_imh(
        jr.PRNGKey(1),
        prior_log_prob_tuple[1],
        prior_log_prob_tuple[0](),
        n_chains=10,
        n_samples=200,
        n_warmup=100,
    )
    chex.assert_shape(samples["mean"], (10, 100, 2))
    chex.assert_shape(samples["std"], (10, 100, 1))
