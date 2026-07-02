# pylint: skip-file

import chex
import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.summary.nass import nass
from sbijax._src.inference.summary.nasss import nasss
from sbijax._src.nn.make_nass_network import make_nass_net, make_nasss_net
from sbijax._src.simulate import simulate


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    noise = tfd.Normal(0.0, 1.0).sample((theta["theta"].shape[0], 4), seed=seed)
    return jnp.tile(theta["theta"], (1, 2)) + noise

  return prior, simulator


# each entry builds a SummaryNet with a summary dimension of 2
SUMMARY_NETS = {
  "nass": lambda: nass(make_nass_net(2, [64, 64])),
  "nasss": lambda: nasss(make_nasss_net(2, 2, [64, 64])),
}


@pytest.mark.parametrize("name", list(SUMMARY_NETS))
def test_fit_then_summarize(name):
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=256)
  sn = SUMMARY_NETS[name]()
  params, info = sn.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=128)
  assert params is not None
  # SummaryNet has no rounds; its Info carries only the loss history (DR-011).
  assert info.losses.ndim == 2 and info.losses.shape[1] == 2
  summaries = sn.summarize(params, data["y"])
  chex.assert_shape(summaries, (256, 2))
