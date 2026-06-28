"""Shared timing / memory / IO helpers for benchmark adapters."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np


def peak_rss_mb() -> float:
  """Peak resident set size of this process, in MB.

  ``ru_maxrss`` is bytes on macOS and kilobytes on Linux.
  """
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if sys.platform == "darwin":
    return rss / (1024 * 1024)
  return rss / 1024


class Timer:
  """Context manager measuring elapsed wall-clock seconds."""

  def __enter__(self) -> "Timer":
    self._t0 = time.perf_counter()
    return self

  def __exit__(self, *exc) -> None:
    self.seconds = time.perf_counter() - self._t0


def save_result(
  out: str,
  label: str,
  seed: int,
  samples: np.ndarray,
  wall_time_s: float,
  compile_time_s: float,
) -> None:
  """Write ``<out>.npy`` (samples) and ``<out>.json`` (metrics)."""
  out_path = Path(out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.save(out_path.with_suffix(".npy"), np.asarray(samples, dtype=np.float64))
  meta = dict(
    label=label,
    seed=seed,
    wall_time_s=float(wall_time_s),
    compile_time_s=float(compile_time_s),
    peak_rss_mb=peak_rss_mb(),
    n_samples=int(np.asarray(samples).shape[0]),
  )
  out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
