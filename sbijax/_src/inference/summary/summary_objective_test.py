import jax.numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.summary.nass import nass
from sbijax._src.nn.make_nass_network import make_nass_net
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train._types import SummaryFns
from sbijax._src.train.fit import fit


def test_nass_summaryfns_trains_and_summarizes():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def sim(seed, theta):
    return jnp.tile(theta["theta"], (1, 4)) + tfd.Normal(0.0, 1.0).sample(
      (theta["theta"].shape[0], 8), seed=seed
    )

  obj = nass(make_nass_net(2, (16, 16)))
  assert isinstance(obj, SummaryFns)
  data = simulate(jr.key(0), prior, sim, n=200)
  params, _ = fit(jr.key(1), obj, data, n_iter=2, batch_size=100)
  s = obj.summarize_fn(params, data["y"])
  assert s.shape[-1] == 2
