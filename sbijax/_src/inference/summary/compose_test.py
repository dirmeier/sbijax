# pylint: skip-file

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.inference.summary._compose import summarized_estimator
from sbijax._src.inference.summary.nass import nass
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.nn.make_nass_network import make_nass_net
from sbijax._src.simulate.simulate import simulate


def _problem():
  # 2-d theta, 4-d data; the summary net reduces the data to 2 dimensions.
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    noise = tfd.Normal(0.0, 1.0).sample((theta["theta"].shape[0], 4), seed=seed)
    return jnp.tile(theta["theta"], (1, 2)) + noise

  return prior, simulator


def test_summarized_estimator_fit_and_sample():
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=256)

  sn = nass(make_nass_net(2, [64, 64]))
  sn_params, _ = sn.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=128)

  # the downstream estimator models the 2-d summary, not the 4-d data
  est = summarized_estimator(nle(prior, make_maf(2)), sn, sn_params)
  params, info = est.fit(jr.PRNGKey(2), data, n_iter=2, batch_size=128)
  assert info.round == 0

  # sample takes the *raw* 4-d observation; the adapter summarizes it
  samples, _ = est.sample(
    jr.PRNGKey(3),
    params,
    jnp.zeros(4),
    n_chains=2,
    n_samples=30,
    n_warmup=10,
  )
  theta = samples["theta"]
  assert theta.ndim == 3 and theta.shape[-1] == 2
