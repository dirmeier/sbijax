"""Compute MCMC reference posteriors (with plots) and run the task inference.

Phase 1 - reference posteriors (true posterior via the TFP slice sampler on each
task's tractable likelihood), stored + plotted:

    python main.py references

Phase 2 - run every task with all 4 SABC algorithms x 5 seeds (20 runs/task),
scored against the stored reference:

    python main.py run [--quick] [--task TASK]

Task definitions live in ``tasks.py``; the four algorithms live in
``adapters/`` and each runs in its own subprocess (clean timing / peak RSS).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import tasks

HERE = Path(__file__).parent
PY = HERE / ".venv" / "bin" / "python"

# (label, adapter module, extra CLI args)
CONFIGS = [
  ("sbijax", "adapters.sbijax_adapter", []),
  ("sabc-mlx", "adapters.mlx_adapter", []),
  ("sabc-numpy", "adapters.numpy_adapter", ["--mode", "numpy"]),
  ("sabc-numba", "adapters.numpy_adapter", ["--mode", "numba"]),
]


def _no_latex():
  """sbijax's import enables text.usetex; force it off (no latex installed)."""
  import matplotlib

  matplotlib.use("Agg")
  matplotlib.rcParams["text.usetex"] = False


# ==========================================================================
# Phase 1: reference posteriors
# ==========================================================================
def _reference_by_chain(name, seed=0, n_chains=4, n_samples=2000, n_warmup=1000):
  from jax import numpy as jnp
  from jax import random as jr
  from sbijax.mcmc import sample_with_slice

  prior, _, likelihood = tasks.build_jax_task(name)
  observed = jnp.asarray(tasks.jax_observed(name))

  def lp(theta):
    return jnp.sum(prior.log_prob(theta)) + jnp.sum(likelihood(observed, theta))

  draws = sample_with_slice(
    jr.PRNGKey(seed), lp, prior,
    n_chains=n_chains, n_samples=n_samples, n_warmup=n_warmup,
  )
  cols = [np.asarray(draws[k]) for k in sorted(draws)]
  return np.concatenate(cols, axis=-1)  # (n_chains, S, n_para)


def _rhat(chains):
  n_chains, s = chains.shape
  if n_chains < 2 or s < 2:
    return float("nan")
  within = chains.var(axis=1, ddof=1).mean()
  between = s * chains.mean(axis=1).var(ddof=1)
  var_hat = (s - 1) / s * within + between / s
  return float(np.sqrt(var_hat / within)) if within > 0 else float("nan")


def _corner(datasets, true, out, title, diag_titles=None):
  """Corner plot: diagonal = 1D KDE, lower triangle = 2D KDE contours.

  ``datasets`` is a list of ``(label, samples, color, fill)``; filled datasets
  (e.g. the reference) are drawn as shaded KDEs, the rest as lines/contours.
  """
  _no_latex()
  import matplotlib.pyplot as plt
  import seaborn as sns

  p = datasets[0][1].shape[1]
  fig, axes = plt.subplots(p, p, figsize=(3 * p, 3 * p), squeeze=False)
  for i in range(p):
    for j in range(p):
      ax = axes[i][j]
      if i == j:
        for _, data, c, fill in datasets:
          sns.kdeplot(x=data[:, i], ax=ax, color=c, lw=1.5, fill=fill,
                      alpha=0.3 if fill else 1.0)
        ax.axvline(true[i], color="k", ls="--", lw=1)
        if diag_titles:
          ax.set_title(diag_titles[i], fontsize=9)
      elif i > j:
        for _, data, c, fill in datasets:
          sns.kdeplot(x=data[:, j], y=data[:, i], ax=ax, color=c, levels=4,
                      fill=fill, alpha=0.4 if fill else 1.0)
        ax.plot(true[j], true[i], "k*", ms=9)
      else:
        ax.axis("off")
      if j == 0 and i > 0:
        ax.set_ylabel(f"theta[{i}]")
      if i == p - 1:
        ax.set_xlabel(f"theta[{j}]")
  handles = [plt.Line2D([], [], color=c, label=lbl) for lbl, _, c, _ in datasets]
  fig.legend(handles=handles, loc="upper right", fontsize=8)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)


def _plot_reference(name, ch, true, out):
  """Reference posterior as a 2D-KDE corner; R-hat per dim in the titles."""
  flat = ch.reshape(-1, ch.shape[-1])
  titles = [f"theta[{d}]  Rhat={_rhat(ch[:, :, d]):.3f}"
            for d in range(ch.shape[-1])]
  _corner([("reference", flat, "C0", True)], true, out,
          f"{name}: slice-sampler reference ({ch.shape[0]} chains)", titles)


def compute_references(seed=0):
  tasks.REF_DIR.mkdir(exist_ok=True)
  for name in tasks.TASK_NAMES:
    ch = _reference_by_chain(name, seed=seed)
    flat = ch.reshape(-1, ch.shape[-1])
    np.save(tasks.REF_DIR / f"{name}.npy", flat)
    np.save(tasks.REF_DIR / f"{name}_observed.npy", tasks.jax_observed(name))
    _plot_reference(name, ch, tasks.true_theta_flat(name),
                    tasks.REF_DIR / f"{name}.png")
    print(f"{name:20s} ref {flat.shape} -> {tasks.REF_DIR / (name + '.npy')} (+png)")


# ==========================================================================
# Phase 2: orchestration + scoring
# ==========================================================================
def _spawn(module, extra, name, seed, quick, out):
  cmd = [str(PY), "-m", module, "--task", name, "--seed", str(seed),
         "--out", out, *extra]
  if quick:
    cmd.append("--quick")
  proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
  if proc.returncode != 0:
    tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
    print(f"  ! {module} [{name} seed={seed}] FAILED:\n{tail}\n")
    return None
  return json.loads(Path(f"{out}.json").read_text())


def _se(x):
  a = np.asarray(x, float)
  return float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def _per_dim(metric, a, b):
  return float(np.mean([metric(a[:, j], b[:, j]) for j in range(a.shape[1])]))


def _benchmark_task(name, n_reps, quick):
  from scipy.stats import energy_distance, wasserstein_distance

  ref_path = tasks.REF_DIR / f"{name}.npy"
  if not ref_path.exists():
    raise SystemExit(f"missing {ref_path}; run `python main.py references` first.")
  ref = np.load(ref_path)
  true = tasks.true_theta_flat(name)

  runs, samples = {}, {}
  for label, module, extra in CONFIGS:
    metas, smps, ok = [], [], True
    for seed in range(n_reps):
      out = str(tasks.RESULTS / f"{name}_{label}_rep{seed}")
      meta = _spawn(module, extra, name, seed, quick, out)
      if meta is None:
        ok = False
        break
      metas.append(meta)
      smps.append(np.load(f"{out}.npy"))
    if ok:
      runs[label], samples[label] = metas, smps
    else:
      print(f"  skipping {label} for {name} (a run failed)")

  if not samples:
    print(f"  no algorithm succeeded for {name}; skipping table/plot")
    return []

  rows = []
  for label in samples:
    walls = [m["wall_time_s"] for m in runs[label]]
    rows.append(dict(
      task=name, label=label,
      wall_mean=float(np.mean(walls)), wall_se=_se(walls),
      compile_mean=float(np.mean([m["compile_time_s"] for m in runs[label]])),
      rss_mean=float(np.mean([m["peak_rss_mb"] for m in runs[label]])),
      bias=float(np.mean([np.linalg.norm(s.mean(0) - true) for s in samples[label]])),
      w1=float(np.mean([_per_dim(wasserstein_distance, s, ref) for s in samples[label]])),
      energy=float(np.mean([_per_dim(energy_distance, s, ref) for s in samples[label]])),
    ))
  _plot_posteriors(name, samples, ref, true)
  return rows


def _plot_posteriors(name, samples, ref, true):
  rng = np.random.default_rng(0)

  def _sub(a, m=4000):
    return a if len(a) <= m else a[rng.choice(len(a), m, replace=False)]

  datasets = [("reference", _sub(ref), "0.4", True)]
  for (label, reps), c in zip(samples.items(), ["C0", "C1", "C2", "C3"]):
    datasets.append((label, _sub(np.concatenate(reps, 0)), c, False))
  _corner(datasets, true, tasks.RESULTS / f"posterior_{name}.png",
          f"{name}: ABC vs reference (2D KDE; dashed/star = true)")


def run(quick=False, task=None):
  tasks.RESULTS.mkdir(exist_ok=True)
  n_reps = 1  # one run per algorithm per task, seed 0
  names = [task] if task else list(tasks.TASK_NAMES)
  all_rows = []
  for name in names:
    rows = _benchmark_task(name, n_reps, quick)
    all_rows.extend(rows)
    if rows:
      print(f"wrote results/posterior_{name}.png")
  (tasks.RESULTS / "metrics.json").write_text(json.dumps(all_rows, indent=2))
  _write_combined(all_rows)
  print("wrote results/table.md (combined)")


def _write_combined(all_rows):
  lines = ["# SABC benchmark (all tasks)", "",
           "Accuracy is vs the slice-sampler reference posterior; wall s",
           "excludes JIT/compile (reported separately).", ""]
  by_task: dict[str, list] = {}
  for r in all_rows:
    by_task.setdefault(r["task"], []).append(r)
  for task_name, rows in by_task.items():
    lines += [f"## {task_name}", "",
              "| algorithm | wall s (mean±se) | compile s | peak RSS MB | "
              "bias L2 | W1 vs ref | energy vs ref |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
      lines.append(
        f"| {r['label']} | {r['wall_mean']:.2f}±{r['wall_se']:.2f} | "
        f"{r['compile_mean']:.2f} | {r['rss_mean']:.0f} | {r['bias']:.4f} | "
        f"{r['w1']:.4f} | {r['energy']:.4f} |"
      )
    lines.append("")
  (tasks.RESULTS / "table.md").write_text("\n".join(lines) + "\n")


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  sub = ap.add_subparsers(dest="cmd", required=True)
  p_ref = sub.add_parser("references", help="phase 1: compute+store+plot")
  p_ref.add_argument("--seed", type=int, default=0)
  p_run = sub.add_parser("run", help="phase 2: 20 runs/task vs reference")
  p_run.add_argument("--quick", action="store_true")
  p_run.add_argument("--task", choices=tasks.TASK_NAMES)

  a = ap.parse_args()
  if a.cmd == "references":
    compute_references(a.seed)
  else:
    run(a.quick, a.task)


if __name__ == "__main__":
  main()
