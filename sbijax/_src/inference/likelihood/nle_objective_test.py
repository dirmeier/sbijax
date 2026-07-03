import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference._sample_info import MCMCSampleInfo
from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.mcmc.nuts import nuts
from sbijax._src.mcmc.sampler import make_sampler
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train._types import ObjectiveFns
from sbijax._src.train.fit import fit
from sbijax._src.train.sample import sample


def test_nle_objective_trains_and_samples_with_sampler():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def sim(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  obj = nle(make_maf(2))
  assert isinstance(obj, ObjectiveFns)
  data = simulate(jr.key(0), prior, sim, n=200)
  params, _ = fit(jr.key(1), obj, data, n_iter=2, batch_size=100)
  sampler = make_sampler(nuts, prior=prior)
  samples, info = sample(
    jr.key(2),
    obj,
    params,
    jnp.zeros(2),
    sampler=sampler,
    n_chains=2,
    n_samples=40,
    n_warmup=20,
  )
  assert samples["theta"].shape == (2, 20, 2)
  assert isinstance(info, MCMCSampleInfo)
