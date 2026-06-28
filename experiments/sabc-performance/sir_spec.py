"""Shared SIR benchmark spec: constants, prior bounds, observed generator."""

from __future__ import annotations

import numpy as np

S0, I0, R0 = 99, 1, 0
N = S0 + I0 + R0
T_MAX = 160.0
MAX_EVENTS = 250

THETA_TRUE = np.array([0.3, 0.1])  # (beta, gamma)
PRIOR_LOW = np.array([0.1, 0.05])
PRIOR_HIGH = np.array([1.0, 0.5])

N_PARA = 2
N_STATS = 3
SUMMARY_NAMES = ("total_infected", "peak_infected", "t_peak")
PARAM_NAMES = ("beta", "gamma")

FULL_BUDGET = dict(n_particles=10_000, n_simulation=1_000_000)
QUICK_BUDGET = dict(n_particles=500, n_simulation=20_000)
N_REPS = 5
OBSERVED_SEED = 12345


def simulate_numpy(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  """Batched Gillespie SIR. theta (B, 2) -> summaries (B, 3).

  Columns of theta are (beta, gamma). Summaries are
  (total_infected, peak_infected, t_peak).
  """
  theta = np.atleast_2d(theta)
  b = theta.shape[0]
  beta, gamma = theta[:, 0], theta[:, 1]
  s = np.full(b, float(S0))
  i = np.full(b, float(I0))
  r = np.zeros(b)
  t = np.zeros(b)
  peak_i = i.copy()
  t_peak = np.zeros(b)
  for _ in range(MAX_EVENTS):
    alive = (i > 0) & (t < T_MAX)
    inf_rate = beta * s * i / N
    rec_rate = gamma * i
    total = inf_rate + rec_rate
    total_safe = np.where(total > 0, total, 1.0)
    dt = -np.log(rng.uniform(size=b)) / total_safe
    is_inf = rng.uniform(size=b) < inf_rate / total_safe
    inf_ev = alive & is_inf
    rec_ev = alive & ~is_inf
    s = s + np.where(inf_ev, -1.0, 0.0)
    i = i + np.where(inf_ev, 1.0, 0.0) + np.where(rec_ev, -1.0, 0.0)
    r = r + np.where(rec_ev, 1.0, 0.0)
    t = t + np.where(alive, dt, 0.0)
    newpeak = alive & (i > peak_i)
    peak_i = np.where(newpeak, i, peak_i)
    t_peak = np.where(newpeak, t, t_peak)
  return np.stack([r, peak_i, t_peak], axis=-1)


def make_observed() -> np.ndarray:
  """Generate the single shared observed summary vector, shape (3,)."""
  rng = np.random.default_rng(OBSERVED_SEED)
  return simulate_numpy(THETA_TRUE.reshape(1, 2), rng).reshape(-1)


if __name__ == "__main__":
  print("observed:", make_observed())
