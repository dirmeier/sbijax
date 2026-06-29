"""sbijax SABC adapter (JAX backend)."""

from __future__ import annotations

import argparse

import numpy as np
from jax import numpy as jnp
from jax import random as jr
from sbijax import SABC, MultiEps, abs_distance, inference_data_as_dictionary

import tasks


def run(name: str, seed: int, budget: dict, out: str) -> None:
  prior, simulator, _ = tasks.build_jax_task(name)
  observed = jnp.asarray(tasks.load_observed(name))
  model = SABC((lambda: prior, simulator), distance_fn=abs_distance)

  def sample_to_numpy(key):
    idata, _ = model.sample_posterior(
      key, observed, n_particles=budget["n_particles"],
      n_simulation=budget["n_simulation"], schedule=MultiEps(v=1.0),
    )
    d = inference_data_as_dictionary(idata.posterior)
    cols = [np.asarray(d[k]).reshape(-1, np.asarray(d[k]).shape[-1])
            for k in sorted(d)]
    return np.concatenate(cols, 1)

  # Warm up JIT and fill the trace cache at the timed budget (compile excluded
  # from wall). ``np.asarray`` inside each block forces materialization, so the
  # timer captures real execution rather than async dispatch.
  with tasks.Timer() as ct:
    sample_to_numpy(jr.PRNGKey(seed))
  with tasks.Timer() as t:
    samples = sample_to_numpy(jr.PRNGKey(seed + 1))
  tasks.save_result(out, "sbijax", seed, samples, t.seconds, ct.seconds)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", required=True, choices=tasks.TASK_NAMES)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--quick", action="store_true")
  p.add_argument("--out", required=True)
  a = p.parse_args()
  run(a.task, a.seed, tasks.QUICK_BUDGET if a.quick else tasks.FULL_BUDGET, a.out)


if __name__ == "__main__":
  main()
