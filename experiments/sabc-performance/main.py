"""Orchestrate the SABC benchmark: run all configs, aggregate, report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import energy_distance, wasserstein_distance

import sir_spec as spec

HERE = Path(__file__).parent
PY = HERE / ".venv" / "bin" / "python"

# (label, module, extra args)
CONFIGS = [
  ("sbijax", "adapters.sbijax_adapter", []),
  ("sabc-mlx", "adapters.mlx_adapter", []),
  ("sabc-numpy", "adapters.numpy_adapter", ["--mode", "numpy"]),
  ("sabc-numba", "adapters.numpy_adapter", ["--mode", "numba"]),
]


def _run_one(module, extra, seed, quick, out) -> dict:
  cmd = [str(PY), "-m", module, "--seed", str(seed), "--out", out, *extra]
  if quick:
    cmd.append("--quick")
  subprocess.run(cmd, cwd=HERE, check=True)
  return json.loads(Path(f"{out}.json").read_text())


def _se(x: list[float]) -> float:
  a = np.asarray(x, dtype=float)
  return float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  args = ap.parse_args()
  n_reps = 1 if args.quick else spec.N_REPS
  budget = spec.QUICK_BUDGET if args.quick else spec.FULL_BUDGET
  results_dir = HERE / "results"
  results_dir.mkdir(exist_ok=True)

  runs: dict[str, list[dict]] = {label: [] for label, _, _ in CONFIGS}
  samples: dict[str, list[np.ndarray]] = {label: [] for label, _, _ in CONFIGS}
  for label, module, extra in CONFIGS:
    for seed in range(n_reps):
      out = str(results_dir / f"{label}_rep{seed}")
      meta = _run_one(module, extra, seed, args.quick, out)
      runs[label].append(meta)
      samples[label].append(np.load(f"{out}.npy"))

  truth = spec.THETA_TRUE
  rows = []
  for label, _, _ in CONFIGS:
    walls = [m["wall_time_s"] for m in runs[label]]
    comps = [m["compile_time_s"] for m in runs[label]]
    rss = [m["peak_rss_mb"] for m in runs[label]]
    bias = np.mean(
      [np.abs(s.mean(axis=0) - truth) for s in samples[label]], axis=0
    )
    # distributional distance vs sabc-numpy, matched per rep.
    w1 = ed = float("nan")
    if label != "sabc-numpy":
      ref = samples["sabc-numpy"]
      w1 = float(np.mean([
        np.mean([
          wasserstein_distance(samples[label][k][:, j], ref[k][:, j])
          for j in range(spec.N_PARA)
        ]) for k in range(n_reps)
      ]))
      ed = float(np.mean([
        np.mean([
          energy_distance(samples[label][k][:, j], ref[k][:, j])
          for j in range(spec.N_PARA)
        ]) for k in range(n_reps)
      ]))
    rows.append(dict(
      label=label,
      wall_mean=float(np.mean(walls)), wall_se=_se(walls),
      compile_mean=float(np.mean(comps)),
      rss_mean=float(np.mean(rss)),
      bias_beta=float(bias[0]), bias_gamma=float(bias[1]),
      w1=w1, ed=ed,
    ))

  (results_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
  _write_table(results_dir / "table.md", rows, n_reps, budget)
  _plot(results_dir / "posterior.png", samples, truth)
  print(f"Wrote {results_dir / 'table.md'} and {results_dir / 'posterior.png'}")


def _write_table(path: Path, rows: list[dict], n_reps: int, budget: dict) -> None:
  lines = [
    f"# SABC benchmark ({n_reps} reps, "
    f"{budget['n_particles']} particles, "
    f"{budget['n_simulation']} sims)",
    "",
    "| config | wall s (mean±se) | compile s | peak RSS MB | "
    "bias beta | bias gamma | W1 vs numpy | energy vs numpy |",
    "|---|---|---|---|---|---|---|---|",
  ]
  for r in rows:
    w1 = "ref" if r["label"] == "sabc-numpy" else f"{r['w1']:.4f}"
    ed = "ref" if r["label"] == "sabc-numpy" else f"{r['ed']:.4f}"
    lines.append(
      f"| {r['label']} | {r['wall_mean']:.2f}±{r['wall_se']:.2f} | "
      f"{r['compile_mean']:.2f} | {r['rss_mean']:.0f} | "
      f"{r['bias_beta']:.4f} | {r['bias_gamma']:.4f} | {w1} | {ed} |"
    )
  path.write_text("\n".join(lines) + "\n")


def _plot(path: Path, samples: dict, truth: np.ndarray) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  names = spec.PARAM_NAMES
  for j in range(spec.N_PARA):
    ax = axes[j]
    for label, reps in samples.items():
      pooled = np.concatenate([r[:, j] for r in reps])
      ax.hist(pooled, bins=50, density=True, histtype="step", label=label)
    ax.axvline(truth[j], color="k", ls="--", lw=1)
    ax.set_title(names[j])
  axes[0].legend(fontsize=8)
  fig.suptitle("SABC posteriors (dashed = truth)")
  fig.tight_layout()
  fig.savefig(path, dpi=120)


if __name__ == "__main__":
  main()
