import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference._sample_info import MCMCSampleInfo
from sbijax._src.mcmc.nuts import nuts
from sbijax._src.mcmc.sampler import make_sampler


def test_make_sampler_draws_from_target():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )
  sampler = make_sampler(nuts, prior=prior)

  def loglik(theta):
    return (
      tfd.Normal(jnp.array([1.0, -1.0]), 1.0).log_prob(theta["theta"]).sum()
    )

  samples, info = sampler(
    jr.key(0), loglik, n_chains=4, n_samples=200, n_warmup=100
  )
  assert samples["theta"].shape == (4, 100, 2)
  assert isinstance(info, MCMCSampleInfo)


def test_make_sampler_chains_move_with_constrained_prior():
  """Chains must explore, not freeze, when a parameter is constrained.

  Regression test: chains used to be initialised from ``N(0, 1)`` on every
  leaf regardless of its support, so a ``HalfNormal`` scale could start
  negative. The target is ``-inf`` there, warmup then adapts a step size of
  zero, and the chain stays pinned at its starting value for every draw —
  producing a "posterior" concentrated on one point per chain.
  """
  prior = tfd.JointDistributionNamed(
    {
      "mean": tfd.Normal(jnp.zeros(2), 1.0),
      "scale": tfd.HalfNormal(jnp.ones(1)),
    },
    batch_ndims=0,
  )
  sampler = make_sampler(nuts, prior=prior)

  def loglik(theta):
    y = jnp.array([-1.0, 1.0])
    return tfd.Normal(theta["mean"], theta["scale"]).log_prob(y).sum()

  n_chains, n_kept = 4, 100
  samples, _ = sampler(
    jr.key(0), loglik, n_chains=n_chains, n_samples=200, n_warmup=100
  )

  # a frozen chain has zero variance and a single unique value.
  for chain in range(n_chains):
    for name in ("mean", "scale"):
      draws = samples[name][chain, :, 0]
      assert jnp.isfinite(draws).all()
      assert draws.std() > 1e-6, f"{name} chain {chain} never moved"
      assert len(set(draws.tolist())) > n_kept // 10

  # a constrained parameter must stay inside its support.
  assert (samples["scale"] > 0).all()
