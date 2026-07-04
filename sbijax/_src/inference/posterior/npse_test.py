# pylint: skip-file

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.experimental._truncated import make_truncated_proposal
from sbijax._src.experimental.nn.make_score_network import make_score_model
from sbijax._src.inference.posterior.npse import npse
from sbijax._src.inference.sequential import run_sequential
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train._types import ObjectiveFns
from sbijax._src.train.sample import sample
from sbijax._src.train.train import train


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


def test_npse_fit_then_sample():
  prior, simulator = _problem()
  obj = npse(make_score_model(2))
  assert isinstance(obj, ObjectiveFns)
  data = simulate(jr.key(0), prior, simulator, n=64)
  params, _ = train(jr.key(1), obj, data, n_iter=2, batch_size=32)
  samples, _ = sample(jr.key(2), obj, params, jnp.zeros(2), n_samples=16)
  theta = samples["theta"]
  assert theta.ndim == 3 and theta.shape[-1] == 2


def test_npse_runs_truncated_sequential():
  prior, simulator = _problem()
  network = make_score_model(2)
  obj = npse(network)
  proposal_fn = make_truncated_proposal(
    prior, network, n_calibration=64, n_prior=1_000
  )
  params, info = run_sequential(
    jr.key(0),
    obj,
    prior,
    simulator,
    jnp.array([-1.0, 1.0]),
    n_rounds=2,
    n_simulations_per_round=32,
    n_iter=2,
    batch_size=32,
    proposal_fn=proposal_fn,
  )
  assert params is not None
  assert info.round == 1
