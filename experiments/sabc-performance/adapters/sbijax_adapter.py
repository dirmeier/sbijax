"""sbijax SABC adapter (JAX backend)."""

from __future__ import annotations

import argparse

import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from sbijax import SABC, MultiEps, abs_distance, inference_data_as_dictionary
from tensorflow_probability.substrates.jax import distributions as tfd

import sir_spec as spec
from adapters import _bench


def _simulate_jax(key, beta, gamma):
  """Batched Gillespie SIR. beta,gamma (B,) -> summaries (B, 3)."""
  b = beta.shape[0]
  s = jnp.full((b,), float(spec.S0))
  i = jnp.full((b,), float(spec.I0))
  r = jnp.zeros((b,))
  t = jnp.zeros((b,))
  peak_i = i
  t_peak = jnp.zeros((b,))
  n = float(spec.N)

  def body(carry, _):
    key, s, i, r, t, peak_i, t_peak = carry
    key, k1, k2 = jr.split(key, 3)
    alive = (i > 0) & (t < spec.T_MAX)
    inf_rate = beta * s * i / n
    rec_rate = gamma * i
    total = inf_rate + rec_rate
    total_safe = jnp.where(total > 0, total, 1.0)
    dt = -jnp.log(jr.uniform(k1, (b,))) / total_safe
    is_inf = jr.uniform(k2, (b,)) < inf_rate / total_safe
    inf_ev = alive & is_inf
    rec_ev = alive & ~is_inf
    s = s + jnp.where(inf_ev, -1.0, 0.0)
    i = i + jnp.where(inf_ev, 1.0, 0.0) + jnp.where(rec_ev, -1.0, 0.0)
    r = r + jnp.where(rec_ev, 1.0, 0.0)
    t = t + jnp.where(alive, dt, 0.0)
    newpeak = alive & (i > peak_i)
    peak_i = jnp.where(newpeak, i, peak_i)
    t_peak = jnp.where(newpeak, t, t_peak)
    return (key, s, i, r, t, peak_i, t_peak), None

  carry = (key, s, i, r, t, peak_i, t_peak)
  (_, s, i, r, t, peak_i, t_peak), _ = jax.lax.scan(
    body, carry, None, length=spec.MAX_EVENTS
  )
  return jnp.stack([r, peak_i, t_peak], axis=-1)


def _prior_fn():
  return tfd.JointDistributionNamed(
    dict(
      theta=tfd.Uniform(
        low=jnp.asarray(spec.PRIOR_LOW), high=jnp.asarray(spec.PRIOR_HIGH)
      )
    ),
    batch_ndims=0,
  )


def _simulator_fn(seed, theta):
  th = theta["theta"]
  beta, gamma = th[..., 0], th[..., 1]
  return _simulate_jax(seed, beta, gamma)


def run(seed: int, budget: dict, out: str) -> None:
  observed = jnp.asarray(spec.make_observed())
  model = SABC((_prior_fn, _simulator_fn), distance_fn=abs_distance)

  def _sample(key, n_particles, n_simulation):
    return model.sample_posterior(
      key,
      observed,
      n_particles=n_particles,
      n_simulation=n_simulation,
      schedule=MultiEps(v=1.0),
    )

  # warm up JIT on the quick budget; record compile time separately.
  with _bench.Timer() as ct:
    idata_w, _ = _sample(
      jr.PRNGKey(seed),
      spec.QUICK_BUDGET["n_particles"],
      spec.QUICK_BUDGET["n_simulation"],
    )
    jax.block_until_ready(idata_w)
  compile_time_s = ct.seconds

  with _bench.Timer() as t:
    idata, _ = _sample(
      jr.PRNGKey(seed + 1), budget["n_particles"], budget["n_simulation"]
    )
    jax.block_until_ready(idata)
  d = inference_data_as_dictionary(idata.posterior)
  theta = np.asarray(d["theta"])  # (chains, draws, 2) -> (N, 2)
  samples = theta.reshape(-1, spec.N_PARA)
  _bench.save_result(out, "sbijax", seed, samples, t.seconds, compile_time_s)


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
