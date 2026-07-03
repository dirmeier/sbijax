# pylint: skip-file

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.diagnostics.sbc import sbc
from sbijax._src.inference.posterior.npe import npe
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train.fit import fit


def _gaussian_problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


def test_sbc_ranks_are_calibrated():
  prior, simulator = _gaussian_problem()
  data = simulate(jr.key(0), prior, simulator, n=2000)
  obj = npe(make_maf(2))
  params, _ = fit(jr.key(1), obj, data, n_iter=500, batch_size=100)

  n_post = 200
  ranks = sbc(
    jr.key(2),
    obj,
    params,
    prior,
    simulator,
    n_simulations=64,
    n_posterior_samples=n_post,
  )
  assert ranks.shape == (64, 2)
  assert jnp.all((ranks >= 0) & (ranks <= n_post))
  # calibrated -> ranks ~ Uniform[0, n_post]: mean ~0.5, std ~ sqrt(1/12).
  normalized = ranks / n_post
  assert jnp.all(jnp.abs(normalized.mean(0) - 0.5) < 0.15)
  assert jnp.all(jnp.abs(normalized.std(0) - jnp.sqrt(1 / 12)) < 0.12)
