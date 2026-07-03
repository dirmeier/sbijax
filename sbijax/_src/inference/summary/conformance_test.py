# pylint: skip-file

import chex
import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.summary.nass import nass
from sbijax._src.inference.summary.nasss import nasss
from sbijax._src.nn.make_nass_network import make_nass_net, make_nasss_net
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train._types import Info, SummaryFns
from sbijax._src.train.fit import fit


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    noise = tfd.Normal(0.0, 1.0).sample((theta["theta"].shape[0], 4), seed=seed)
    return jnp.tile(theta["theta"], (1, 2)) + noise

  return prior, simulator


# each entry builds a SummaryFns with a summary dimension of 2
SUMMARY_NETS = {
  "nass": lambda: nass(make_nass_net(2, [64, 64])),
  "nasss": lambda: nasss(make_nasss_net(2, 2, [64, 64])),
}


@pytest.mark.parametrize("name", list(SUMMARY_NETS))
def test_fit_then_summarize(name):
  prior, simulator = _problem()
  data = simulate(jr.key(0), prior, simulator, n=256)
  obj = SUMMARY_NETS[name]()
  assert isinstance(obj, SummaryFns)
  params, info = fit(jr.key(1), obj, data, n_iter=2, batch_size=128)
  assert params is not None
  # SummaryFns share the generic Info; assert the loss history shape (DR-011).
  assert isinstance(info, Info)
  assert info.losses.ndim == 2 and info.losses.shape[1] == 2
  summaries = obj.summarize_fn(params, data["y"])
  chex.assert_shape(summaries, (256, 2))
