# pylint: skip-file

import jax
import optax
import pytest
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.experimental.nn.make_score_network import make_score_model
from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.inference.likelihood.snle import snle
from sbijax._src.inference.posterior.fmpe import fmpe
from sbijax._src.inference.posterior.npe import npe
from sbijax._src.inference.posterior.npse import npse
from sbijax._src.inference.ratio.nre import nre
from sbijax._src.mcmc.nuts import nuts
from sbijax._src.mcmc.sampler import make_sampler
from sbijax._src.nn.make_continuous_flow import make_cnf
from sbijax._src.nn.make_flow import make_maf, make_spf
from sbijax._src.nn.make_mlp import make_mlp
from sbijax._src.simulate.simulate import simulate
from sbijax._src.train._types import Info, ObjectiveFns, TrainingState
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


def _batch(data, n=32):
  """A single training batch matching what the dataloader feeds ``init_fn``.

  The dataloader flattens the named ``theta`` pytree to a flat ``(n, d)``
  array per row, so the toy batch does the same.
  """
  theta = jax.vmap(lambda x: ravel_pytree(x)[0])(data["theta"])
  return {"y": data["y"][:n], "theta": theta[:n]}


# registry of trainable objectives to check against the ObjectiveFns contract.
# each entry builds an ObjectiveFns from a prior and records whether sampling
# requires an injected MCMC sampler (nle/snle/nre) or is amortized (npe/fmpe/
# npse).
ESTIMATORS = {
  "npe": {"build": lambda p: npe(make_maf(2)), "mcmc": False},
  "npe_spf": {"build": lambda p: npe(make_spf(2, -5.0, 5.0)), "mcmc": False},
  "fmpe": {"build": lambda p: fmpe(make_cnf(2)), "mcmc": False},
  "npse": {"build": lambda p: npse(make_score_model(2)), "mcmc": False},
  "nle": {"build": lambda p: nle(make_maf(2)), "mcmc": True},
  "snle": {"build": lambda p: snle(make_maf(2)), "mcmc": True},
  "nre": {"build": lambda p: nre(make_mlp()), "mcmc": True},
}


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_fit_returns_params_and_info(name):
  prior, simulator = _problem()
  data = simulate(jr.key(0), prior, simulator, n=200)
  obj = ESTIMATORS[name]["build"](prior)
  assert isinstance(obj, ObjectiveFns)
  params, info = train(jr.key(1), obj, data, n_iter=2, batch_size=100)
  assert params is not None
  # structural Info contract (DR-011): every Info exposes an int ``round`` and
  # a ``(n_epochs, 2)`` train/validation loss history.
  assert isinstance(info, Info) and info.round == 0
  assert info.losses.ndim == 2 and info.losses.shape[1] == 2


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_primitive_seam(name):
  prior, simulator = _problem()
  data = simulate(jr.key(0), prior, simulator, n=200)
  obj = ESTIMATORS[name]["build"](prior)
  batch = _batch(data)
  optimizer = optax.adam(1e-3)
  state = obj.train.init_fn(optimizer, jr.key(1), batch)
  assert isinstance(state, TrainingState)
  metrics, state2 = obj.train.step_fn(optimizer, jr.key(2), state, batch)
  assert isinstance(metrics, dict) and "loss" in metrics
  assert isinstance(state2, TrainingState)
  assert isinstance(obj.train.eval_fn(jr.key(3), state, batch), dict)


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_sample_returns_named_pytree_and_info(name):
  prior, simulator = _problem()
  data = simulate(jr.key(0), prior, simulator, n=200)
  obj = ESTIMATORS[name]["build"](prior)
  params, _ = train(jr.key(1), obj, data, n_iter=2, batch_size=100)
  if ESTIMATORS[name]["mcmc"]:
    sampler = make_sampler(nuts, prior=prior)
    kwargs = {"n_chains": 2, "n_samples": 30, "n_warmup": 10}
  else:
    sampler = None
    kwargs = {"n_samples": 64}
  samples, info = sample(
    jr.key(2), obj, params, jnp.zeros(2), sampler=sampler, **kwargs
  )
  theta = samples["theta"]
  assert theta.ndim == 3 and theta.shape[-1] == 2  # (n_chains, n_draws, dim)
  # structural (pytree, record) contract (DR-012): info is a NamedTuple.
  assert isinstance(info, tuple) and hasattr(info, "_fields")


def test_npe_extra_is_objective():
  prior, _ = _problem()
  assert isinstance(npe(make_maf(2)).extra(prior), ObjectiveFns)
