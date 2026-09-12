"""Posterior recovery against conjugate models with analytic posteriors.

Every other inference test in the package trains for ``n_iter=2`` and asserts
on shapes: they pin the ``ObjectiveFns`` contract, not correctness. These tests
fit each estimator properly on a conjugate problem whose posterior is known in
closed form, and compare every marginal against the exact CDF.

They are marked ``slow`` and deselected from the default run; see the
``recovery`` job in ``.github/workflows/ci.yaml``.
"""

from typing import Any, NamedTuple

import jax
import pytest
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.experimental.nn.make_score_network import make_score_model
from sbijax._src.inference.abc.sabc import sabc
from sbijax._src.inference.abc.smcabc import smcabc
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
from sbijax._src.train.sample import sample
from sbijax._src.train.train import train

# training budget shared by every neural estimator. 10_000 simulations matches
# the getting-started notebook; early stopping usually halts well before 200.
N_SIMULATIONS = 10_000
N_ITER = 200
N_EARLY_STOPPING_PATIENCE = 20
N_POSTERIOR_SAMPLES = 4_000

# MCMC budget for the likelihood- and ratio-based methods.
MCMC_KWARGS = {"n_chains": 4, "n_samples": 2_000, "n_warmup": 1_000}

Y_OBS = jnp.array([-1.0, 1.0])


class Marginal(NamedTuple):
  """One scalar parameter coordinate and its exact posterior marginal.

  Attributes:
      key: the leaf of the prior's pytree holding this component
      component: the coordinate within that leaf
      dist: the exact posterior marginal, a scalar ``tfd`` distribution
  """

  key: str
  component: int
  dist: Any

  @property
  def name(self):
    """A label for assertion messages, e.g. ``"mean[0]"``."""
    return f"{self.key}[{self.component}]"


class Problem(NamedTuple):
  """A conjugate inference problem with a known posterior.

  Attributes:
      prior: the prior over parameters
      simulator: a callable ``(rng_key, theta) -> y``
      y_obs: the single observation to condition on
      n_theta: the flattened parameter dimension, sizing posterior networks
      n_y: the data dimension, sizing likelihood networks
      marginals: the exact posterior marginal for every parameter coordinate
  """

  prior: Any
  simulator: Any
  y_obs: Any
  n_theta: int
  n_y: int
  marginals: list[Marginal]


def _normal_problem():
  """Conjugate Gaussian: ``theta ~ N(0, I)``, ``y | theta ~ N(theta, I)``."""
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )

  def simulator(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["theta"]), 1.0)
    return theta["theta"] + p.sample(seed=seed)

  # unit prior precision plus unit likelihood precision, so the posterior
  # precision is 2: theta | y ~ N(y / 2, 1 / 2).
  posterior = tfd.Normal(Y_OBS / 2.0, jnp.sqrt(0.5))
  marginals = [Marginal("theta", i, posterior[i]) for i in range(2)]
  return Problem(prior, simulator, Y_OBS, 2, 2, marginals)


# prior hyperparameters of the getting-started notebook's NIG model.
A0, B0, KAPPA0 = 3.0, 2.0, 1.0


def _nig_problem():
  """Normal-inverse-gamma model taken from the getting-started notebook.

  ``variance ~ InvGamma(a0, b0)``, ``mean | variance ~ N(0, variance / k0)``
  and ``y | mean, variance ~ N(mean, variance)``. Conjugate, so the posterior
  is again normal-inverse-gamma with the marginals below.
  """
  # batch_ndims=0 is not in the notebook's version of this prior, but without
  # it log_prob is shaped (n, 2) rather than (n,), which the ABC engines
  # broadcast against per-particle quantities and fail on.
  prior = tfd.JointDistributionNamed(
    {
      "variance": tfd.InverseGamma(
        concentration=A0 * jnp.ones(1), scale=B0 * jnp.ones(1)
      ),
      "mean": lambda variance: tfd.Normal(
        jnp.zeros(2), jnp.sqrt(variance / KAPPA0)
      ),
    },
    batch_ndims=0,
  )

  def simulator(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["mean"]), jnp.sqrt(theta["variance"]))
    return theta["mean"] + p.sample(seed=seed)

  # conjugate update for a single observation with prior mean zero
  kappa1 = KAPPA0 + 1.0
  mu1 = Y_OBS / kappa1
  a1 = A0 + 1.0
  b1 = B0 + 0.5 * (KAPPA0 / kappa1) * jnp.sum(Y_OBS**2)
  variance_dist = tfd.InverseGamma(concentration=a1, scale=b1)
  # marginalising the variance out of the normal leaves a Student-t
  mean_dist = tfd.StudentT(
    df=2.0 * a1, loc=mu1, scale=jnp.sqrt(b1 / (a1 * kappa1))
  )
  marginals = [
    Marginal("mean", 0, mean_dist[0]),
    Marginal("mean", 1, mean_dist[1]),
    Marginal("variance", 0, variance_dist),
  ]
  return Problem(prior, simulator, Y_OBS, 3, 2, marginals)


PROBLEMS = {"normal": _normal_problem, "nig": _nig_problem}


def _l2(x, y):
  return jax.vmap(jnp.linalg.norm)(x - y)


# registry of estimators. ``build`` takes the whole problem because the network
# dimension differs by family: posterior estimators are sized by the parameter
# dimension, likelihood estimators by the data dimension, and nre's classifier
# is dimension-agnostic. ``kind`` selects the execution path in
# ``_posterior_draws``.
ESTIMATORS = {
  "npe": {
    "build": lambda p: npe(make_maf(p.n_theta)),
    "kind": "amortized",
  },
  "npe_spf": {
    "build": lambda p: npe(make_spf(p.n_theta, -5.0, 5.0)),
    "kind": "amortized",
  },
  "fmpe": {
    "build": lambda p: fmpe(make_cnf(p.n_theta)),
    "kind": "amortized",
  },
  "npse": {
    "build": lambda p: npse(make_score_model(p.n_theta)),
    "kind": "amortized",
  },
  "nle": {
    "build": lambda p: nle(make_maf(p.n_y)),
    "kind": "mcmc",
  },
  "snle": {
    "build": lambda p: snle(make_maf(p.n_y)),
    "kind": "mcmc",
  },
  "nre": {
    "build": lambda p: nre(make_mlp()),
    "kind": "mcmc",
  },
  "sabc": {
    "build": lambda p: sabc(p.prior, p.simulator),
    "kind": "abc",
    # SABC carries an unweighted population, so its particles are draws
    "sample_kwargs": {"n_particles": 2_000, "n_simulation": 100_000},
  },
  "smcabc": {
    "build": lambda p: smcabc(p.prior, p.simulator, lambda x: x, _l2),
    "kind": "abc",
    # ess_min sits just below n_particles so the final round resamples: the
    # engine returns particles without their log-weights, and an unweighted
    # ECDF of weighted particles would not estimate the posterior.
    "sample_kwargs": {
      "n_rounds": 10,
      "n_particles": 5_000,
      "ess_min": 4_500,
    },
  },
}


def _posterior_draws(rng_key, problem, spec):
  """Fit the estimator where needed and return draws as a named pytree."""
  if spec["kind"] == "abc":
    sampler = spec["build"](problem)
    draws, _ = sampler.sample(rng_key, problem.y_obs, **spec["sample_kwargs"])
    return draws

  sim_key, train_key, sample_key = jr.split(rng_key, 3)
  objective = spec["build"](problem)
  data = simulate(sim_key, problem.prior, problem.simulator, n=N_SIMULATIONS)
  params, _ = train(
    train_key,
    objective,
    data,
    n_iter=N_ITER,
    n_early_stopping_patience=N_EARLY_STOPPING_PATIENCE,
  )
  if spec["kind"] == "mcmc":
    sampler = make_sampler(nuts, prior=problem.prior)
    kwargs = MCMC_KWARGS
  else:
    sampler = None
    kwargs = {"n_samples": N_POSTERIOR_SAMPLES}
  draws, _ = sample(
    sample_key,
    objective,
    params,
    problem.y_obs,
    sampler=sampler,
    prior=problem.prior,
    **kwargs,
  )
  return draws


def _ks_distance(samples, dist):
  """Kolmogorov-Smirnov distance between samples and an exact marginal.

  The amortized estimators model an unconstrained parameter vector, so they
  put some mass below zero for the NIG variance. ``tfd`` returns nan for a cdf
  evaluated below its support where the true value is 0; substituting it keeps
  those draws counting against the fit instead of poisoning the statistic.
  Every marginal used here is unbounded above, so nan can only mean "below".
  """
  x = jnp.sort(samples)
  n = x.shape[0]
  cdf = jnp.nan_to_num(dist.cdf(x), nan=0.0)
  return jnp.maximum(
    jnp.max(jnp.arange(1, n + 1) / n - cdf),
    jnp.max(cdf - jnp.arange(n) / n),
  )


def recovery_errors(rng_key, problem_name, method_name):
  """Return per-marginal ``(ks, mean_error)`` pairs for one combination.

  Exposed so tolerances can be recalibrated without going through pytest.

  Args:
      rng_key: a jax random key
      problem_name: a key of ``PROBLEMS``
      method_name: a key of ``ESTIMATORS``

  Returns:
      a list of ``(ks_distance, absolute_mean_error)`` tuples, one per marginal
  """
  problem = PROBLEMS[problem_name]()
  draws = _posterior_draws(rng_key, problem, ESTIMATORS[method_name])
  errors = []
  for marginal in problem.marginals:
    x = draws[marginal.key][..., marginal.component].reshape(-1)
    ks = float(_ks_distance(x, marginal.dist))
    mean_error = float(jnp.abs(jnp.mean(x) - marginal.dist.mean()))
    errors.append((ks, mean_error))
  return errors


# (ks, mean) tolerances, set at roughly twice the worst value measured over
# seeds 0-2, with the measurement in a trailing comment. The neural and MCMC
# methods land an order of magnitude inside these; sabc and npse are the loose
# ones and are the entries to watch if this file ever turns flaky.
TOLERANCES = {
  ("normal", "npe"): (0.09, 0.14),  # measured 0.041, 0.065
  ("normal", "npe_spf"): (0.11, 0.18),  # measured 0.052, 0.088
  ("normal", "fmpe"): (0.08, 0.12),  # measured 0.035, 0.055
  ("normal", "npse"): (0.19, 0.36),  # measured 0.095, 0.176
  ("normal", "nle"): (0.08, 0.09),  # measured 0.040, 0.044
  ("normal", "snle"): (0.08, 0.09),  # measured 0.040, 0.044
  ("normal", "nre"): (0.07, 0.09),  # measured 0.034, 0.044
  ("normal", "sabc"): (0.23, 0.34),  # measured 0.113, 0.169
  ("normal", "smcabc"): (0.07, 0.09),  # measured 0.034, 0.044
  ("nig", "npe"): (0.10, 0.15),  # measured 0.047, 0.073
  ("nig", "npe_spf"): (0.13, 0.15),  # measured 0.062, 0.074
  ("nig", "fmpe"): (0.17, 0.10),  # measured 0.084, 0.048
  ("nig", "npse"): (0.33, 0.50),  # measured 0.162, 0.250
  ("nig", "nle"): (0.08, 0.10),  # measured 0.036, 0.050
  ("nig", "snle"): (0.08, 0.10),  # measured 0.036, 0.050
  ("nig", "nre"): (0.08, 0.11),  # measured 0.038, 0.055
  ("nig", "sabc"): (0.38, 0.72),  # measured 0.186, 0.359
  ("nig", "smcabc"): (0.07, 0.05),  # measured 0.035, 0.021
}

CASES = [(p, m) for p in PROBLEMS for m in ESTIMATORS]


@pytest.mark.slow
@pytest.mark.parametrize(("problem_name", "method_name"), CASES)
def test_recovers_analytic_posterior(problem_name, method_name):
  problem = PROBLEMS[problem_name]()
  draws = _posterior_draws(jr.key(0), problem, ESTIMATORS[method_name])
  # every method must name its draws after the prior, not hand back a flat
  # vector; sample(prior=...) is what makes the amortized ones comply
  assert set(draws) == set(problem.prior.event_shape)
  ks_tol, mean_tol = TOLERANCES[(problem_name, method_name)]
  for marginal in problem.marginals:
    x = draws[marginal.key][..., marginal.component].reshape(-1)
    # the distributional check: the whole marginal, not just its moments
    ks = _ks_distance(x, marginal.dist)
    assert ks < ks_tol, f"{marginal.name}: ks={ks:.4f} exceeds {ks_tol}"
    # redundant given the KS check, but names the failure in familiar units
    mean_error = jnp.abs(jnp.mean(x) - marginal.dist.mean())
    assert mean_error < mean_tol, (
      f"{marginal.name}: posterior mean off by {mean_error:.4f}, "
      f"got {jnp.mean(x):.4f} want {marginal.dist.mean():.4f}"
    )
