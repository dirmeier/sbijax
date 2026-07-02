# pylint: skip-file

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.experimental._truncated import make_truncated_proposal
from sbijax._src.experimental.nn.make_score_network import make_score_model
from sbijax._src.experimental.npse import npse
from sbijax._src.inference.sequential import run_sequential
from sbijax._src.simulate import simulate


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
  data = simulate(jr.PRNGKey(0), prior, simulator, n=64)
  est = npse(prior, make_score_model(2))
  params, _ = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=32)
  idata = est.sample(jr.PRNGKey(2), params, jnp.zeros(2), n_samples=16)
  theta = idata["/posterior"]["theta"].data
  assert theta.ndim == 3 and theta.shape[-1] == 2


def test_npse_runs_truncated_sequential():
  prior, simulator = _problem()
  network = make_score_model(2)
  est = npse(prior, network)
  proposal_fn = make_truncated_proposal(
    prior, network, n_calibration=64, n_prior=1_000
  )
  params, info = run_sequential(
    jr.PRNGKey(0),
    est,
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
