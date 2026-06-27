# Design: Port SABC (Simulated Annealing ABC) to sbijax

- **Date:** 2026-06-27
- **Status:** Approved (design); pending implementation plan
- **Branch:** `sabc`
- **Reference implementation:** `sabc-mlx` (Python + C++/MLX), `sabc.run`
- **Paper:** Albert, Ulzega, Dirmeier, Scheidegger, Bassi, Mira,
  *"Simulated Annealing ABC with multiple summary statistics"*,
  arXiv:2505.23261 (2025). (arXiv title: *"A Thermodynamic Approach to ABC
  with Multiple Summary Statistics"*.)

## 1. Goal

Add a JAX-native, jittable implementation of Simulated Annealing ABC (SABC) to
sbijax as a new ABC method, with **full parity** to the `sabc-mlx` reference:
both the single-epsilon and multi-epsilon (one temperature per summary
statistic) schedules, the Differential-Evolution MCMC kernel, the empirical-CDF
distance transform, and importance resampling.

SABC is the algorithm from the paper; sbijax currently ships only `SMCABC`
(Beaumont 2009 adaptive SMC-ABC), so the paper's method is absent from the
JAX library.

Non-goals (deferred to a future commit): unifying the ABC API surface
(`SMCABC` vs `SABC` constructor/return conventions). For now the priority is a
correct, working JAX implementation that follows existing sbijax patterns.

## 2. Public API

New module `sbijax/_src/abc/sabc.py`. Exports added to `sbijax/__init__.py`:
`SABC`, `SingleEps`, `MultiEps`, `DiffEvolution`, `abs_distance`,
`sq_distance`, `weighted_sq`.

```python
class SABC(SBI):
  def __init__(self, model_fns, summary_fn=lambda x: x,
               distance_fn=abs_distance): ...

  def sample_posterior(
      self, rng_key, observable,
      n_particles=1000,
      n_simulation=100_000,
      schedule=SingleEps(v=1.0),
      proposal=DiffEvolution(),
      delta=0.1,
  ) -> (InferenceData, sabc_info): ...
```

- Same constructor shape as `SMCABC` (`model_fns, summary_fn, distance_fn`).
- **Contract difference vs SMCABC:** `distance_fn(summary_simulated,
  summary_observed)` must return a **per-statistic** array of shape
  `(n_particles, n_stats)` — it must *not* collapse to a scalar norm. The
  per-statistic axis is essential: the multi-epsilon schedule assigns one
  temperature per statistic.
- Returns an ArviZ `InferenceData` (via `as_inference_data`, sbijax
  convention) plus a `sabc_info` namedtuple carrying the traces
  `epsilon_history`, `u_history`, and the final raw distances `rho`.

### Built-in distance callables

Ship the three reference transforms so users don't have to write their own.
Each operates elementwise on `(summary_simulated - summary_observed)`,
preserving the statistic axis:

```python
abs_distance(s, o)      # |s - o|                         -> (B, n_stats)
sq_distance(s, o)       # (s - o) ** 2                     -> (B, n_stats)
weighted_sq(weights)    # factory -> distance_fn computing (s - o)**2 * weights
```

`weighted_sq` is a factory returning a closure that carries the `weights`
array, so the resulting callable keeps the standard `(s, o)` signature.

### Config classes (ported 1:1 from sabc-mlx)

Frozen dataclasses with `__post_init__` validation:

```python
@dataclass(frozen=True)
class SingleEps:   # one shared epsilon
  v: float = 1.0   # annealing speed, must be > 0

@dataclass(frozen=True)
class MultiEps:    # one epsilon per summary statistic
  v: float = 1.0   # annealing speed, must be > 0

@dataclass(frozen=True)
class DiffEvolution:
  gamma0: float | None = None  # None -> 2.38 / sqrt(2 * n_para)
  sigma_gamma: float = 1e-5    # multiplicative jitter on gamma
```

(The `algorithm` string field from the MLX version is dropped; schedule
dispatch is by Python type, not by string.)

## 3. Internal structure

All inference logic lives in module-level **pure, jittable** functions, each
with one concern. `SABC.sample_posterior` is a thin orchestrator that ravels
the prior, calls the jitted core, and wraps the result.

| Function | Concern |
|----------|---------|
| `_distance(summary_fn, distance_fn, y, ss_obs)` | `(B, n_stats)` distances |
| `_build_cdf(rho)` | knot tables `(values, probs)`, shape `(B+2, n_stats)` |
| `_cdf_eval(tables, rho)` | empirical-CDF transform `u`, `(B, n_stats)` |
| `_resample_weights(u, delta)` | importance weights `(B,)` |
| `_resample_indices(u, delta, key)` | categorical resample indices |
| `_epsilon_single(ubar, v)` | scalar root-find for shared epsilon |
| `_epsilon_multi(u, v)` | per-statistic root-finds (vmapped) |
| `_de_propose(theta, donors, gamma0, sigma_gamma, key)` | DE-MCMC proposal |
| `_bisect(f, lo, hi, n_iter)` | fixed-iteration vectorized root finder |
| `_step(carry, _)` | one annealing iteration (the `lax.scan` body) |

### Prior raveling

Reuse `jax.flatten_util.ravel_pytree` + `jax.vmap` (as `SMCABC` already does)
to flatten the `tfd.JointDistributionNamed` prior into `(N, n_para)` and back.
This replaces sabc-mlx's `FlatPrior` adapter. `rvs(key, size)` samples the
prior and ravels; `logpdf(theta)` unravels and calls `prior.log_prob`.

## 4. JAX-specific replacements

The C++ core does two things JAX cannot copy directly.

### 4.1 Root finder (replaces Brent)

`_bisect(f, lo, hi, n_iter=60)` — a fixed-iteration vectorized bisection,
fully jittable, no bracketing-failure branch (the bracket is known to contain
the root for both epsilon functions). 60 iterations over the relevant brackets
converge to ~1e-12, matching Brent for these monotone functions.

- `_epsilon_single`: solve `eps^2 + v*eps^1.5 - ubar^2 = 0` for
  `eps in [0, ubar]`. Returns `0` when `ubar <= 1e-12`.
- `_epsilon_multi`: keeps the `cn` / `num` / `den` constant computation from
  the reference and is `jax.vmap`ped over the statistic axis. Per statistic,
  solve the `g(beta) = ubar` equation on a bracket `[1e-6, bhi]` with the
  series-expansion guard for small `beta`, then assemble `eps`.

Both are validated numerically against the `sabc-mlx` / scipy `brentq` results
in tests (see §7).

### 4.2 Annealing loop (replaces the Python for-loop)

`lax.scan` over `n_updates = n_simulation // n_particles`.

- **Carry:** `(population, u, rho, logprior, epsilon, key)`.
- **Scan constants:** the CDF knot tables (`values`, `probs`), built **once**
  from the initial prior draw and held fixed for the whole run (matches the
  reference: `build_cdf` at init, `cdf_eval` thereafter).
- **Per step:**
  1. `inv_eps = 1 / epsilon`.
  2. Two **sequential** half-batch DE-MCMC updates (exact port of
     `update_half`): half 1 uses half 2 as the frozen donor pool; half 2 then
     sees the updated half 1. Acceptance log-ratio:
     `lp_prop - lp_cur + sum((u_cur - u_prop) * inv_eps)`, gated on
     `isfinite(lp_prop)`.
  3. Conditional resample via `lax.cond` on the reference's every-2-iterations
     cadence (`resample_every = 2 * n_particles`).
  4. Recompute `epsilon` from the current `u` (single or multi).
- **Scan outputs (stacked into histories):** `epsilon` per step and
  `mean(u, axis=0)` per step. The initial epsilon and initial mean-u
  (post initial-resample) are prepended, matching the reference.

## 5. Data flow

```
prior.sample (ravel)
  -> _distance               (B, n_stats) raw rho
  -> _build_cdf              once, fixed knot tables
  -> _cdf_eval               u
  -> initial _resample        on u
  -> initial epsilon
  -> lax.scan(_step)          n_updates iterations
  -> unravel final population
  -> as_inference_data + sabc_info(epsilon_history, u_history, rho)
```

## 6. Numerical-fidelity notes

- **CDF knot tables:** replicate the reference's behavior of *not*
  de-duplicating zeros/ties (`np.unique` is intentionally omitted there for
  fixed-shape arrays). SABC distances are continuous, so this matches in
  practice; documenting it keeps the port faithful and shape-stable for jit.
- **Interp:** `_cdf_eval` uses `jnp.searchsorted` + clamped linear
  interpolation, equivalent to the C++ `interp_1d` (count of knots `<= q`,
  clamped to `[1, K-1]`, interpolate within the bracketing interval).
- **DE partner indices:** distinct partners drawn as in the reference
  (`i2 in [0, M-1)` shifted by `>= i1` to skip `i1`).
- **dtypes:** sbijax runs float32 by default; epsilon root-finds compute in
  float (matching the reference's float32 `u`). No float64 promotion required.

## 7. Testing & validation

New `sbijax/_src/abc/sabc_test.py`:

- **Recovery smoke tests** on the existing 2D-Gaussian `prior_simulator_tuple`
  fixture (per the "validate via simple generative examples" preference):
  posterior mean recovers the observed value, for **both** `SingleEps` and
  `MultiEps`.
- **Jit test:** the core inference function compiles and runs under
  `jax.jit` end-to-end.
- **Root-finder tests:** `_epsilon_single` / `_epsilon_multi` agree with
  scipy `brentq` on the reference equations to a tight tolerance.
- **Shape tests:** `epsilon_history`, `u_history` lengths
  (`n_updates + 1`) and the returned `InferenceData` structure.

## 8. Documentation

- Google-style docstrings on `SABC`, the config classes, and the distance
  callables.
- Add `SABC` to `docs/sbijax.rst`.
- Add the Albert et al. 2025 citation to `docs/references.bib` and cite it in
  the `SABC` docstring (mirroring how `SMCABC` cites Beaumont 2009).

## 9. Files touched

| File | Change |
|------|--------|
| `sbijax/_src/abc/sabc.py` | new — `SABC`, configs, distances, core fns |
| `sbijax/_src/abc/sabc_test.py` | new — tests |
| `sbijax/__init__.py` | export new public symbols |
| `docs/sbijax.rst` | document `SABC` |
| `docs/references.bib` | add Albert et al. 2025 |
