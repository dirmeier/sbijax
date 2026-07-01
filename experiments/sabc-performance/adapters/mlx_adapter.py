"""sabc-mlx adapter (MLX backend). Model lives in tasks.build_mlx_task."""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
import sabc
import tasks


def run(name: str, seed: int, budget: dict, out: str) -> None:
  """Run sabc-mlx on task ``name`` and write samples + timing to ``out``."""
  mx.random.seed(seed)
  observed = mx.array(tasks.load_observed(name))
  prior, simulate = tasks.build_mlx_task(name)

  def f_dist(theta, obs):
    return mx.abs(simulate(theta) - obs)

  with tasks.Timer() as t:
    post = sabc.run(
      f_dist,
      prior=prior,
      observed=observed,
      n_particles=budget["n_particles"],
      n_simulation=budget["n_simulation"],
      schedule=sabc.MultiEps(v=1.0),
      proposal=sabc.DiffEvolution(),
      key=mx.random.key(seed),
    )
    mx.eval(post.samples)
  tasks.save_result(
    out, "sabc-mlx", seed, np.asarray(post.samples), t.seconds, 0.0
  )


def main() -> None:
  """Parse CLI arguments and run the adapter."""
  p = argparse.ArgumentParser()
  p.add_argument("--task", required=True, choices=tasks.TASK_NAMES)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--quick", action="store_true")
  p.add_argument("--out", required=True)
  a = p.parse_args()
  run(
    a.task, a.seed, tasks.QUICK_BUDGET if a.quick else tasks.FULL_BUDGET, a.out
  )


if __name__ == "__main__":
  main()
