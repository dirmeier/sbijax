# pylint: skip-file

import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.inference.likelihood.snle import snle
from sbijax._src.inference.posterior.cmpe import cmpe
from sbijax._src.inference.posterior.fmpe import fmpe
from sbijax._src.inference.posterior.npe import npe
from sbijax._src.inference.ratio.nre import nre
from sbijax._src.nn.make_consistency_model import make_cm
from sbijax._src.nn.make_continuous_flow import make_cnf
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.nn.make_mlp import make_mlp
from sbijax._src.simulate.simulate import simulate


def _problem():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    return theta["theta"] + tfd.Normal(0.0, 1.0).sample(
      theta["theta"].shape, seed=seed
    )

  return prior, simulator


# registry of trainable estimators to check against the Estimator contract.
# each entry builds the estimator from a prior and provides the sample kwargs
# appropriate to its sampling style (MCMC-based vs direct).
ESTIMATORS = {
  "nle": {
    "build": lambda prior: nle(prior, make_maf(2)),
    "sample_kwargs": {"n_chains": 2, "n_samples": 30, "n_warmup": 10},
  },
  "fmpe": {
    "build": lambda prior: fmpe(prior, make_cnf(2)),
    "sample_kwargs": {"n_samples": 64},
  },
  "npe": {
    "build": lambda prior: npe(prior, make_maf(2)),
    "sample_kwargs": {"n_samples": 64},
  },
  "cmpe": {
    "build": lambda prior: cmpe(prior, make_cm(2)),
    "sample_kwargs": {"n_samples": 64},
  },
  "nre": {
    "build": lambda prior: nre(prior, make_mlp()),
    "sample_kwargs": {"n_chains": 2, "n_samples": 30, "n_warmup": 10},
  },
  "snle": {
    "build": lambda prior: snle(prior, make_maf(2)),
    "sample_kwargs": {"n_chains": 2, "n_samples": 30, "n_warmup": 10},
  },
}


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_fit_returns_params_and_info(name):
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=200)
  est = ESTIMATORS[name]["build"](prior)
  params, info = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=100)
  assert params is not None
  # structural Info contract (DR-011): every per-method Info exposes an int
  # `round` and a (n_epochs, 2) `losses` history.
  assert isinstance(info.round, int) and info.round == 0
  assert info.losses.ndim == 2 and info.losses.shape[1] == 2


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_fit_advances_round_when_info_passed(name):
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=200)
  est = ESTIMATORS[name]["build"](prior)
  _, info0 = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=100)
  _, info1 = est.fit(jr.PRNGKey(1), data, info=info0, n_iter=2, batch_size=100)
  assert info1.round == 1


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_sample_returns_named_pytree_and_info(name):
  prior, simulator = _problem()
  data = simulate(jr.PRNGKey(0), prior, simulator, n=200)
  est = ESTIMATORS[name]["build"](prior)
  params, _ = est.fit(jr.PRNGKey(1), data, n_iter=2, batch_size=100)
  samples, info = est.sample(
    jr.PRNGKey(2), params, jnp.zeros(2), **ESTIMATORS[name]["sample_kwargs"]
  )
  theta = samples["theta"]
  assert theta.ndim == 3 and theta.shape[-1] == 2  # (n_chains, n_draws, dim)
  # structural (pytree, record) contract (DR-012): info is a NamedTuple
  assert isinstance(info, tuple) and hasattr(info, "_fields")
