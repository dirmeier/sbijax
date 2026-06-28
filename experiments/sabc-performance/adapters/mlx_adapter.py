"""sabc-mlx adapter (MLX backend)."""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
import sabc
from sabc import distributions as dist

import sir_spec as spec
from adapters import _bench


def _simulate_mlx(theta: mx.array) -> mx.array:
  """Batched Gillespie SIR. theta (B, 2) -> summaries (B, 3)."""
  b = theta.shape[0]
  beta, gamma = theta[:, 0], theta[:, 1]
  s = mx.full((b,), float(spec.S0))
  i = mx.full((b,), float(spec.I0))
  r = mx.zeros((b,))
  t = mx.zeros((b,))
  peak_i = i
  t_peak = mx.zeros((b,))
  n = float(spec.N)
  for _ in range(spec.MAX_EVENTS):
    alive = mx.logical_and(i > 0, t < spec.T_MAX)
    inf_rate = beta * s * i / n
    rec_rate = gamma * i
    total = inf_rate + rec_rate
    total_safe = mx.where(total > 0, total, 1.0)
    dt = -mx.log(mx.random.uniform(shape=(b,))) / total_safe
    is_inf = mx.random.uniform(shape=(b,)) < inf_rate / total_safe
    inf_ev = mx.logical_and(alive, is_inf)
    rec_ev = mx.logical_and(alive, mx.logical_not(is_inf))
    s = s + mx.where(inf_ev, -1.0, 0.0)
    i = i + mx.where(inf_ev, 1.0, 0.0) + mx.where(rec_ev, -1.0, 0.0)
    r = r + mx.where(rec_ev, 1.0, 0.0)
    t = t + mx.where(alive, dt, 0.0)
    newpeak = mx.logical_and(alive, i > peak_i)
    peak_i = mx.where(newpeak, i, peak_i)
    t_peak = mx.where(newpeak, t, t_peak)
  return mx.stack([r, peak_i, t_peak], axis=-1)


# Compile the 250-event loop once into a fused kernel (MLX's analogue of
# jax.jit). The global RNG state must be threaded as a captured input/output,
# otherwise mx.compile freezes the random draws and every batch is identical.
_simulate_compiled = mx.compile(
  _simulate_mlx, inputs=mx.random.state, outputs=mx.random.state
)


def run(seed: int, budget: dict, out: str) -> None:
  mx.random.seed(seed)
  observed = mx.array(spec.make_observed().astype(np.float64))

  def f_dist(theta: mx.array, obs: mx.array) -> mx.array:
    return mx.abs(_simulate_compiled(theta) - obs)

  prior = dist.JointDistributionNamed(
    {
      "theta": dist.Uniform(
        mx.array(spec.PRIOR_LOW.tolist()),
        mx.array(spec.PRIOR_HIGH.tolist()),
      )
    }
  )
  with _bench.Timer() as t:
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
  _bench.save_result(
    out, "sabc-mlx", seed, np.asarray(post.samples), t.seconds, 0.0
  )


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--quick", action="store_true")
  p.add_argument("--out", required=True)
  a = p.parse_args()
  budget = spec.QUICK_BUDGET if a.quick else spec.FULL_BUDGET
  run(a.seed, budget, a.out)


if __name__ == "__main__":
  main()
