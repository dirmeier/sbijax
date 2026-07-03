# pylint: skip-file

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.inference.summary._compose import summarized_estimator
from sbijax._src.inference.summary.nass import nass
from sbijax._src.mcmc.nuts import nuts
from sbijax._src.mcmc.sampler import make_sampler
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.nn.make_nass_network import make_nass_net
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train._types import ObjectiveFns
from sbijax._src.train.fit import fit
from sbijax._src.train.sample import sample


def _problem():
  # 2-d theta, 4-d data; the summary net reduces the data to 2 dimensions.
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    noise = tfd.Normal(0.0, 1.0).sample(
      (theta["theta"].shape[0], 4), seed=seed
    )
    return jnp.tile(theta["theta"], (1, 2)) + noise

  return prior, simulator


def test_summarized_estimator_fit_and_sample():
  prior, simulator = _problem()
  data = simulate(jr.key(0), prior, simulator, n=256)

  sn = nass(make_nass_net(2, [64, 64]))
  sn_params, _ = fit(jr.key(1), sn, data, n_iter=2, batch_size=128)

  # the downstream estimator models the 2-d summary, not the 4-d data
  est = summarized_estimator(nle(make_maf(2)), sn, sn_params)
  assert isinstance(est, ObjectiveFns)
  params, _ = fit(jr.key(2), est, data, n_iter=2, batch_size=128)

  # sample takes the *raw* 4-d observation; the adapter summarizes it
  samples, _ = sample(
    jr.key(3),
    est,
    params,
    jnp.zeros(4),
    sampler=make_sampler(nuts, prior=prior),
    n_chains=2,
    n_samples=30,
    n_warmup=10,
  )
  assert samples["theta"].shape == (2, 20, 2)
