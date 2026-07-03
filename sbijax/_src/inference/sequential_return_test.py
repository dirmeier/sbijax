import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.posterior.npe import npe
from sbijax._src.inference.sequential import run_sequential
from sbijax._src.nn.make_flow import make_maf


def test_run_sequential_npe_completes_with_pytree_proposal():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  params, info = run_sequential(
    jr.PRNGKey(0),
    npe(prior, make_maf(2)),
    prior,
    simulator,
    jnp.zeros(2),
    n_rounds=2,
    n_simulations_per_round=100,
    n_iter=2,
    batch_size=100,
  )
  assert info.round == 1
