"""Benchmark tasks.

Backend imports (jax / mlx / numba) are lazy inside the ``build_*`` functions so
a subprocess only loads the backend it measures, keeping per-process RSS honest;
``PLC0415`` is silenced file-wide for that reason.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REF_DIR = HERE / "reference_posteriors"
RESULTS = HERE / "results"

QUICK_BUDGET = {"n_particles": 500, "n_simulation": 20_000}
FULL_BUDGET = {"n_particles": 10_000, "n_simulation": 1_000_000}
N_REPS = 5
OBSERVED_SEED = 23

TASK_NPARA = {"gaussian_mixture": 6, "mixture_distractors": 1, "two_moons": 2}
TASK_NAMES = tuple(TASK_NPARA)

TM_ANG = -math.pi / 4.0
TM_COS, TM_SIN = math.cos(TM_ANG), math.sin(TM_ANG)
TM_BASE, TM_RLOC, TM_RSCALE = 0.25, 0.1, 0.01
TM_ALOW, TM_AHIGH = -math.pi / 2.0, math.pi / 2.0

MD_ALPHA, MD_SIGMA, MD_NDIS = 0.3, 0.3, 8
GAUSSIAN_MIXTURE_MEAN = np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
GAUSSIAN_MIXTURE_COVS = np.array(
  [
    [[0.7, 0.0], [0.0, 0.05]],
    [[0.7, 0.0], [0.0, 0.05]],
    [[1.0, 0.95], [0.95, 1.0]],
  ]
)
GAUSSIAN_MIXTURE_CHOL = np.linalg.cholesky(GAUSSIAN_MIXTURE_COVS)  # (3, 2, 2)

TRUE_THETA = {
  "gaussian_mixture": GAUSSIAN_MIXTURE_MEAN.reshape(1, 6),
  "mixture_distractors": np.array([[1.5]]),
  "two_moons": np.array([[0.0, 0.0]]),
}
FIXED_OBSERVED = {"two_moons": np.zeros(2)}


def true_theta_flat(name: str) -> np.ndarray:
  """Return the data-generating parameter for ``name`` as a flat array."""
  return TRUE_THETA[name].reshape(-1)


def load_observed(name: str) -> np.ndarray:
  """Load the fixed observation stored for ``name`` by ``main.py references``."""
  return np.load(REF_DIR / f"{name}_observed.npy").astype(np.float64)


def peak_rss_mb() -> float:
  """Peak resident set size of this process in MB (platform-corrected)."""
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


class Timer:
  """Context manager measuring elapsed wall-clock seconds."""

  def __enter__(self):
    """Start the timer."""
    self._t0 = time.perf_counter()
    return self

  def __exit__(self, *exc):
    """Stop the timer and store the elapsed time in ``seconds``."""
    self.seconds = time.perf_counter() - self._t0


def save_result(out, label, seed, samples, wall_s, compile_s) -> None:  # noqa: PLR0913
  """Write samples (``<out>.npy``) and timing/memory metrics (``<out>.json``)."""
  out_path = Path(out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.save(out_path.with_suffix(".npy"), np.asarray(samples, dtype=np.float64))
  out_path.with_suffix(".json").write_text(
    json.dumps(
      {
        "label": label,
        "seed": seed,
        "wall_time_s": float(wall_s),
        "compile_time_s": float(compile_s),
        "peak_rss_mb": peak_rss_mb(),
        "n_samples": int(np.asarray(samples).shape[0]),
      },
      indent=2,
    )
  )


def build_jax_task(name: str):
  """Return ``(prior, simulator, likelihood)`` in JAX for ``name``."""
  import jax
  from jax import numpy as jnp
  from jax import random as jr
  from tensorflow_probability.substrates.jax import distributions as tfd

  if name == "two_moons":
    from sbijax._src.simulators.two_moons import two_moons

    return two_moons()
  if name == "mixture_distractors":
    from sbijax._src.simulators.mixture_model_distractors import (
      mixture_model_with_distractors,
    )

    _, mixture_distractors_simulator, mixture_distractors_likelihood = (
      mixture_model_with_distractors()
    )
    prior = tfd.JointDistributionNamed(
      {
        "theta": tfd.Independent(
          tfd.Uniform(jnp.array([-10.0]), jnp.array([10.0])), 1
        )
      }
    )
    return prior, mixture_distractors_simulator, mixture_distractors_likelihood
  if name != "gaussian_mixture":
    raise KeyError(name)

  mean, covs, chol = (
    jnp.asarray(GAUSSIAN_MIXTURE_MEAN),
    jnp.asarray(GAUSSIAN_MIXTURE_COVS),
    jnp.asarray(GAUSSIAN_MIXTURE_CHOL),
  )

  def prior_fn():
    return tfd.JointDistributionNamed(
      {"theta": tfd.Independent(tfd.Normal(mean, jnp.ones(6)), 1)}
    )

  def gaussian_mixture_simulator(seed, theta):
    theta = theta["theta"].reshape(-1, 6)
    n = theta.shape[0]
    c_key, e_key = jr.split(seed)
    idx = tfd.Categorical(probs=jnp.full(3, 1 / 3)).sample(
      seed=c_key, sample_shape=(n,)
    )
    means = theta.reshape(n, 3, 2)
    mean_sel = means[jnp.arange(n), idx]
    eps = tfd.Normal(0.0, 1.0).sample(seed=e_key, sample_shape=(n, 2))
    return mean_sel + jnp.einsum("bij,bj->bi", chol[idx], eps)

  def gaussian_mixture_likelihood(y, theta):
    y = y.reshape(-1, 2)
    means = theta["theta"].reshape(-1, 6).reshape(-1, 3, 2)
    lps = [
      jnp.log(1 / 3)
      + tfd.MultivariateNormalFullCovariance(means[:, k, :], covs[k]).log_prob(
        y
      )
      for k in range(3)
    ]
    return jax.scipy.special.logsumexp(jnp.stack(lps, 0), axis=0)

  return prior_fn(), gaussian_mixture_simulator, gaussian_mixture_likelihood


def jax_observed(name: str) -> np.ndarray:
  """Return the fixed observation for ``name`` (canonical point or simulated)."""
  if name in FIXED_OBSERVED:
    return FIXED_OBSERVED[name].astype(np.float64)
  from jax import numpy as jnp
  from jax import random as jr

  _, simulator, _ = build_jax_task(name)
  true = {"theta": jnp.asarray(TRUE_THETA[name])}
  return np.asarray(simulator(jr.PRNGKey(OBSERVED_SEED), true)).reshape(-1)


def build_mlx_task(name: str):
  """Return ``(prior, simulate)`` in MLX for ``name``."""
  import mlx.core as mx
  from sabc import distributions as dist

  if name == "two_moons":
    prior = dist.JointDistributionNamed(
      {"theta": dist.Uniform(mx.array([-1.0, -1.0]), mx.array([1.0, 1.0]))}
    )

    def two_moons_sim(theta):
      theta = theta.reshape(-1, 2)
      n = theta.shape[0]
      a = mx.random.uniform(TM_ALOW, TM_AHIGH, shape=(n, 1))
      r = TM_RLOC + TM_RSCALE * mx.random.normal(shape=(n, 1))
      p = mx.concatenate([mx.cos(a) * r + TM_BASE, mx.sin(a) * r], axis=1)
      z0 = TM_COS * theta[:, 0:1] - TM_SIN * theta[:, 1:2]
      z1 = TM_SIN * theta[:, 0:1] + TM_COS * theta[:, 1:2]
      return p + mx.concatenate([-mx.abs(z0), z1], axis=1)

    return prior, two_moons_sim

  if name == "mixture_distractors":
    prior = dist.JointDistributionNamed(
      {"theta": dist.Uniform(mx.array([-10.0]), mx.array([10.0]))}
    )

    def mixture_distractors_sim(theta):
      theta = theta.reshape(-1, 1)
      n = theta.shape[0]
      param = mx.broadcast_to(theta, (n, 2))
      pick = mx.random.uniform(shape=(n, 2)) < MD_ALPHA
      mean = mx.where(pick, param, -param)
      scale = mx.where(pick, 1.0, MD_SIGMA)
      y = mean + mx.random.normal(shape=(n, 2)) * scale
      return mx.concatenate([y, mx.random.normal(shape=(n, MD_NDIS))], axis=1)

    return prior, mixture_distractors_sim

  prior = dist.JointDistributionNamed(
    {"theta": dist.Normal(mx.array(GAUSSIAN_MIXTURE_MEAN.tolist()), mx.ones(6))}
  )
  chol = mx.array(GAUSSIAN_MIXTURE_CHOL)

  def gaussian_mixture_sim(theta):
    theta = theta.reshape(-1, 6)
    n = theta.shape[0]
    means = theta.reshape(n, 3, 2)
    idx = mx.random.categorical(mx.zeros((n, 3)))
    onehot = (idx.reshape(n, 1) == mx.arange(3).reshape(1, 3)).astype(
      mx.float32
    )
    mean_sel = (onehot.reshape(n, 3, 1) * means).sum(axis=1)
    chol_sel = (onehot.reshape(n, 3, 1, 1) * chol.reshape(1, 3, 2, 2)).sum(
      axis=1
    )
    eps = mx.random.normal(shape=(n, 2))
    return mean_sel + (chol_sel * eps.reshape(n, 1, 2)).sum(axis=2)

  return prior, gaussian_mixture_sim


class UniformPrior:
  """Uniform prior with a numpy ``rvs``/``logpdf`` interface."""

  def __init__(self, low, high):
    """Store the (low, high) bounds and the log-volume normaliser."""
    self.low, self.high = np.asarray(low, float), np.asarray(high, float)
    self._logvol = float(np.sum(np.log(self.high - self.low)))

  def rvs(self, rng, size=1):
    """Draw ``size`` samples, shape ``(size, n_para)``."""
    return rng.uniform(self.low, self.high, size=(size, self.low.size))

  def logpdf(self, theta):
    """Log density of ``theta`` ``(N, n_para)`` -> ``(N,)``."""
    theta = np.atleast_2d(theta)
    ok = np.all((theta >= self.low) & (theta <= self.high), axis=1)
    lp = np.full(theta.shape[0], -np.inf)
    lp[ok] = -self._logvol
    return lp


class NormalPrior:
  """Independent-normal prior with a numpy ``rvs``/``logpdf`` interface."""

  def __init__(self, mean, std):
    """Store the per-dimension mean and standard deviation."""
    self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)

  def rvs(self, rng, size=1):
    """Draw ``size`` samples, shape ``(size, n_para)``."""
    return self.mean + self.std * rng.standard_normal((size, self.mean.size))

  def logpdf(self, theta):
    """Log density of ``theta`` ``(N, n_para)`` -> ``(N,)``."""
    z = (np.atleast_2d(theta) - self.mean) / self.std
    return np.sum(-0.5 * z**2 - np.log(self.std) - 0.5 * np.log(2 * np.pi), 1)


def _np_prior(name):
  if name == "gaussian_mixture":
    return NormalPrior(GAUSSIAN_MIXTURE_MEAN, np.ones(6))
  if name == "mixture_distractors":
    return UniformPrior([-10.0], [10.0])
  return UniformPrior([-1.0, -1.0], [1.0, 1.0])


def _two_moons_np(theta, y, rng):
  n = theta.shape[0]
  a = rng.uniform(TM_ALOW, TM_AHIGH, size=(n, 1))
  r = TM_RLOC + TM_RSCALE * rng.standard_normal((n, 1))
  p = np.concatenate([np.cos(a) * r + TM_BASE, np.sin(a) * r], axis=1)
  z0 = TM_COS * theta[:, 0:1] - TM_SIN * theta[:, 1:2]
  z1 = TM_SIN * theta[:, 0:1] + TM_COS * theta[:, 1:2]
  y[:] = p + np.concatenate([-np.abs(z0), z1], axis=1)


def _mixture_distractors_np(theta, y, rng):
  n = theta.shape[0]
  param = np.broadcast_to(theta.reshape(n, 1), (n, 2))
  pick = rng.uniform(size=(n, 2)) < MD_ALPHA
  y[:, :2] = np.where(pick, param, -param) + rng.standard_normal(
    (n, 2)
  ) * np.where(pick, 1.0, MD_SIGMA)
  y[:, 2:] = rng.standard_normal((n, MD_NDIS))


def _gaussian_mixture_np(theta, y, rng):
  n = theta.shape[0]
  idx = rng.integers(0, 3, size=n)
  means = theta.reshape(n, 3, 2)
  eps = rng.standard_normal((n, 2))
  y[:] = means[np.arange(n), idx] + np.einsum(
    "bij,bj->bi", GAUSSIAN_MIXTURE_CHOL[idx], eps
  )


def _stats_np(y, ss_out):
  ss_out[:] = y


_NP_SIMS = {
  "gaussian_mixture": _gaussian_mixture_np,
  "mixture_distractors": _mixture_distractors_np,
  "two_moons": _two_moons_np,
}


def build_numpy_task(name: str):
  """Return ``(prior, np_sim, nb_sim, stats_np, stats_nb)`` for ``name``."""
  import numba as nb

  @nb.njit(cache=True)
  def two_moons_nb(theta, y):
    a = np.random.uniform(TM_ALOW, TM_AHIGH)
    r = TM_RLOC + TM_RSCALE * np.random.standard_normal()
    z0 = TM_COS * theta[0] - TM_SIN * theta[1]
    z1 = TM_SIN * theta[0] + TM_COS * theta[1]
    y[0] = math.cos(a) * r + TM_BASE - abs(z0)
    y[1] = math.sin(a) * r + z1

  @nb.njit(cache=True)
  def mixture_distractors_nb(theta, y):
    for k in range(2):
      if np.random.random() < MD_ALPHA:
        m, sc = theta[0], 1.0
      else:
        m, sc = -theta[0], MD_SIGMA
      y[k] = m + sc * np.random.standard_normal()
    for k in range(2, 2 + MD_NDIS):
      y[k] = np.random.standard_normal()

  @nb.njit(cache=True)
  def gaussian_mixture_nb(theta, y):
    idx = min(int(np.random.random() * 3.0), 2)
    e0, e1 = np.random.standard_normal(), np.random.standard_normal()
    L = GAUSSIAN_MIXTURE_CHOL[idx]
    y[0] = theta[2 * idx] + L[0, 0] * e0 + L[0, 1] * e1
    y[1] = theta[2 * idx + 1] + L[1, 0] * e0 + L[1, 1] * e1

  @nb.njit(cache=True)
  def stats_nb(y, ss):
    for i in range(ss.shape[0]):
      ss[i] = y[i]

  nb_sims = {
    "gaussian_mixture": gaussian_mixture_nb,
    "mixture_distractors": mixture_distractors_nb,
    "two_moons": two_moons_nb,
  }
  return _np_prior(name), _NP_SIMS[name], nb_sims[name], _stats_np, stats_nb
