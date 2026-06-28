"""sabc-numpy / sabc-numba adapter (simulated_annealing_abc)."""

from __future__ import annotations

import argparse
import os

import numba as nb
import numpy as np
from simulated_annealing_abc import (
  DifferentialEvolution,
  SABCConfig,
  make_f_dist,
  sabc,
)

import sir_spec as spec
from adapters import _bench

# Module-level constants so Numba can freeze them inside the njit kernel
# (attribute access like ``spec.S0`` is not reliable inside @njit).
_S0 = float(spec.S0)
_I0 = float(spec.I0)
_N = float(spec.N)
_T_MAX = float(spec.T_MAX)
_MAX_EVENTS = int(spec.MAX_EVENTS)


class UniformPrior:
  """Uniform prior over (beta, gamma) with the rvs/logpdf batch API."""

  def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
    self.low = np.asarray(low, dtype=np.float64)
    self.high = np.asarray(high, dtype=np.float64)
    self._logvol = float(np.sum(np.log(self.high - self.low)))

  def rvs(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    return rng.uniform(self.low, self.high, size=(size, self.low.size))

  def logpdf(self, theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(theta)
    in_bounds = np.all((theta >= self.low) & (theta <= self.high), axis=1)
    lp = np.full(theta.shape[0], -np.inf)
    lp[in_bounds] = -self._logvol
    return lp


def _simulator_numpy(theta: np.ndarray, y: np.ndarray, rng) -> None:
  """Batch: fill y (B, 3) with SIR summaries for theta (B, 2)."""
  y[:] = spec.simulate_numpy(theta, rng)


def _stats_identity(y: np.ndarray, ss_out: np.ndarray) -> None:
  """Summaries already computed by the simulator; copy through."""
  ss_out[:] = y


@nb.njit(cache=True)
def _simulator_nb(theta, y):
  """Single-particle Gillespie SIR. theta (2,) -> y (3,)."""
  beta = theta[0]
  gamma = theta[1]
  s = _S0
  i = _I0
  r = 0.0
  t = 0.0
  peak_i = i
  t_peak = 0.0
  for _ in range(_MAX_EVENTS):
    alive = (i > 0.0) and (t < _T_MAX)
    inf_rate = beta * s * i / _N
    rec_rate = gamma * i
    total = inf_rate + rec_rate
    total_safe = total if total > 0.0 else 1.0
    dt = -np.log(np.random.random()) / total_safe
    is_inf = np.random.random() < inf_rate / total_safe
    if alive and is_inf:
      s -= 1.0
      i += 1.0
    elif alive and (not is_inf):
      i -= 1.0
      r += 1.0
    if alive:
      t += dt
    if alive and (i > peak_i):
      peak_i = i
      t_peak = t
  y[0] = r
  y[1] = peak_i
  y[2] = t_peak


@nb.njit(cache=True)
def _stats_nb(y, ss):
  ss[0] = y[0]
  ss[1] = y[1]
  ss[2] = y[2]


def run(mode: str, seed: int, budget: dict, out: str) -> None:
  ss_obs = spec.make_observed()
  prior = UniformPrior(spec.PRIOR_LOW, spec.PRIOR_HIGH)
  n_workers = os.cpu_count() or 1
  compile_time_s = 0.0

  if mode == "numba":
    f_dist = make_f_dist(
      n_samples=spec.N_STATS,
      ss_obs=ss_obs,
      use_numba=True,
      n_workers=n_workers,
      simulator=_simulator_nb,
      stats_fn=_stats_nb,
    )
    # warm up the njit kernels; record compile time, exclude from wall time.
    with _bench.Timer() as ct:
      tmp = np.empty((1, spec.N_STATS), dtype=np.float64)
      f_dist(prior.rvs(np.random.default_rng(0), 1), out=tmp)
    compile_time_s = ct.seconds
  else:
    f_dist = make_f_dist(
      n_samples=spec.N_STATS,
      ss_obs=ss_obs,
      simulator=_simulator_numpy,
      stats_fn=_stats_identity,
      seed=seed,
      n_workers=n_workers,
    )

  config = SABCConfig(
    f_dist=f_dist,
    prior=prior,
    n_particles=budget["n_particles"],
    v=1.0,
    algorithm="multi_eps",
    proposal=DifferentialEvolution(n_para=spec.N_PARA),
    seed=seed,
    show_progressbar=False,
    show_checkpoint=None,
  )
  with _bench.Timer() as t:
    result = sabc(config, n_simulation=budget["n_simulation"])
  _bench.save_result(
    out, f"sabc-{mode}", seed, result.population, t.seconds, compile_time_s
  )


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--mode", choices=["numpy", "numba"], required=True)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--quick", action="store_true")
  p.add_argument("--out", required=True)
  a = p.parse_args()
  budget = spec.QUICK_BUDGET if a.quick else spec.FULL_BUDGET
  run(a.mode, a.seed, budget, a.out)


if __name__ == "__main__":
  main()
