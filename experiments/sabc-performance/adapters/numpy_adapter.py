"""sabc-numpy / sabc-numba adapter. Model lives in tasks.build_numpy_task."""

from __future__ import annotations

import argparse
import os

import numpy as np
from simulated_annealing_abc import (
  DifferentialEvolution,
  SABCConfig,
  make_f_dist,
  sabc,
)

import tasks


def run(name: str, mode: str, seed: int, budget: dict, out: str) -> None:
  prior, np_sim, nb_sim, stats_np, stats_nb = tasks.build_numpy_task(name)
  ss_obs = tasks.load_observed(name)
  obs_dim = ss_obs.size
  n_workers = os.cpu_count() or 1
  compile_s = 0.0

  if mode == "numba":
    f_dist = make_f_dist(
      n_samples=obs_dim, ss_obs=ss_obs, use_numba=True, n_workers=n_workers,
      simulator=nb_sim, stats_fn=stats_nb,
    )
    with tasks.Timer() as ct:  # warm up njit (excluded from wall time)
      f_dist(prior.rvs(np.random.default_rng(0), 1), out=np.empty((1, obs_dim)))
    compile_s = ct.seconds
  else:
    f_dist = make_f_dist(
      n_samples=obs_dim, ss_obs=ss_obs, simulator=np_sim, stats_fn=stats_np,
      seed=seed, n_workers=n_workers,
    )

  config = SABCConfig(
    f_dist=f_dist, prior=prior, n_particles=budget["n_particles"], v=1.0,
    algorithm="multi_eps", proposal=DifferentialEvolution(n_para=tasks.TASK_NPARA[name]),
    seed=seed, show_progressbar=False, show_checkpoint=None,
  )
  with tasks.Timer() as t:
    result = sabc(config, n_simulation=budget["n_simulation"])
  tasks.save_result(out, f"sabc-{mode}", seed, result.population, t.seconds, compile_s)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", required=True, choices=tasks.TASK_NAMES)
  p.add_argument("--mode", choices=["numpy", "numba"], required=True)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--quick", action="store_true")
  p.add_argument("--out", required=True)
  a = p.parse_args()
  run(a.task, a.mode, a.seed,
      tasks.QUICK_BUDGET if a.quick else tasks.FULL_BUDGET, a.out)


if __name__ == "__main__":
  main()
