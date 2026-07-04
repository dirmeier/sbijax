import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.inference.posterior.npe import npe
from sbijax._src.inference.sequential import run_sequential
from sbijax._src.mcmc.nuts import nuts
from sbijax._src.mcmc.sampler import make_sampler
from sbijax._src.nn.make_flow import make_maf


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def sim(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, sim


def test_run_sequential_npe_advances_round():
  prior, sim = _problem()
  params, info = run_sequential(
    jr.key(0),
    npe(make_maf(2)),
    prior,
    sim,
    jnp.zeros(2),
    n_rounds=2,
    n_simulations_per_round=100,
    n_iter=2,
    batch_size=100,
  )
  assert info.round == 1


def test_run_sequential_nle_with_sampler():
  prior, sim = _problem()
  params, info = run_sequential(
    jr.key(0),
    nle(make_maf(2)),
    prior,
    sim,
    jnp.zeros(2),
    n_rounds=2,
    n_simulations_per_round=100,
    sampler=make_sampler(nuts, prior=prior),
    n_iter=2,
    batch_size=100,
  )
  assert info.round == 1
