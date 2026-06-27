# SABC Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **COMMIT POLICY (user override):** Do **not** run `git commit` unless the
> user explicitly asks. The "Commit" steps below mean: stage the changes
> (`git add ...`), then **stop and ask the user** whether to commit. Never
> commit autonomously.

**Goal:** Add a JAX-native, jittable Simulated Annealing ABC (`SABC`) method to
sbijax with full parity to the `sabc-mlx` reference (single/multi epsilon
schedules, DE-MCMC kernel, empirical-CDF transform, importance resampling).

**Architecture:** One new module `sbijax/_src/abc/sabc.py` holding the public
`SABC` class (subclass of `SBI`, mirroring `SMCABC`), the config dataclasses,
the built-in distance callables, and module-level pure jittable functions. The
annealing loop is a `lax.scan`; epsilon root-finds use a fixed-iteration
vectorized bisection. The prior is raveled to `(N, n_para)` via
`jax.flatten_util.ravel_pytree` and unraveled to the named dict only when
calling the simulator.

**Tech Stack:** JAX (`jax`, `jax.numpy`, `jax.random`, `jax.lax`),
`tensorflow_probability.substrates.jax` priors, ArviZ `InferenceData` via
sbijax's `as_inference_data`. Tests with `pytest`; epsilon cross-checks with
`scipy.optimize.brentq`.

**Reference:** Albert et al. 2025, arXiv:2505.23261. Spec:
`docs/superpowers/specs/2026-06-27-port-sabc-to-sbijax-design.md`.

**Run all commands from `/Users/simon/PROJECTS/sbijax`.** Use `uv run pytest`.

---

## File structure

| File | Responsibility |
|------|----------------|
| `sbijax/_src/abc/sabc.py` | new — `SABC`, configs, distances, all pure fns |
| `sbijax/_src/abc/sabc_test.py` | new — unit + recovery tests |
| `sbijax/__init__.py` | export `SABC`, configs, distance callables |
| `docs/references.bib` | add Albert et al. 2025 entry |
| `docs/sbijax.rst` | document `SABC` |

---

## Task 1: Config classes and distance callables

**Files:**
- Create: `sbijax/_src/abc/sabc.py`
- Test: `sbijax/_src/abc/sabc_test.py`

- [ ] **Step 1: Write the failing test**

Create `sbijax/_src/abc/sabc_test.py`:

```python
# pylint: skip-file

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.abc.sabc import (
    DiffEvolution,
    MultiEps,
    SingleEps,
    abs_distance,
    sq_distance,
    weighted_sq,
)


def test_single_eps_rejects_nonpositive_v():
    with pytest.raises(ValueError):
        SingleEps(v=0.0)


def test_multi_eps_rejects_nonpositive_v():
    with pytest.raises(ValueError):
        MultiEps(v=-1.0)


def test_diff_evolution_defaults():
    de = DiffEvolution()
    assert de.gamma0 is None
    assert de.sigma_gamma == 1e-5


def test_distance_callables_preserve_stat_axis():
    s = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    o = jnp.array([0.0, 0.0])
    assert abs_distance(s, o).shape == (2, 2)
    np.testing.assert_allclose(abs_distance(s, o), jnp.abs(s))
    np.testing.assert_allclose(sq_distance(s, o), s**2)
    w = jnp.array([1.0, 0.5])
    np.testing.assert_allclose(weighted_sq(w)(s, o), s**2 * w)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (sabc.py missing).

- [ ] **Step 3: Write minimal implementation**

Create `sbijax/_src/abc/sabc.py`:

```python
r"""Simulated Annealing ABC (SABC).

JAX port of the ``sabc`` reference implementation. Implements the algorithm
from :cite:t:`albert2025simulated`.

References:
    Albert, Carlo, et al. "Simulated Annealing ABC with multiple summary
    statistics". arXiv preprint arXiv:2505.23261, 2025.
"""

from collections import namedtuple
from dataclasses import dataclass

import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src._sbi_base import SBI
from sbijax._src.util.data import as_inference_data

_CDF_INFLATE = 1.5


@dataclass(frozen=True)
class SingleEps:
  """Single shared epsilon schedule.

  Args:
      v: annealing speed; must be positive.
  """

  v: float = 1.0

  def __post_init__(self):
    if self.v <= 0:
      raise ValueError(f"v must be positive, got {self.v}.")


@dataclass(frozen=True)
class MultiEps:
  """One epsilon per summary statistic.

  Args:
      v: annealing speed; must be positive.
  """

  v: float = 1.0

  def __post_init__(self):
    if self.v <= 0:
      raise ValueError(f"v must be positive, got {self.v}.")


@dataclass(frozen=True)
class DiffEvolution:
  """Differential-Evolution proposal configuration.

  Args:
      gamma0: base DE step; if ``None`` the core uses ``2.38/sqrt(2*n_para)``.
      sigma_gamma: relative Gaussian jitter on the step size.
  """

  gamma0: float | None = None
  sigma_gamma: float = 1e-5


def abs_distance(simulated, observed):
  """Absolute per-statistic distance ``|simulated - observed|``."""
  return jnp.abs(simulated - observed)


def sq_distance(simulated, observed):
  """Squared per-statistic distance ``(simulated - observed) ** 2``."""
  return jnp.square(simulated - observed)


def weighted_sq(weights):
  """Return a weighted squared per-statistic distance callable.

  Args:
      weights: per-statistic weights, shape ``(n_stats,)``.

  Returns:
      A callable ``(simulated, observed) -> (B, n_stats)``.
  """
  weights = jnp.asarray(weights)

  def _fn(simulated, observed):
    return jnp.square(simulated - observed) * weights

  return _fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Stage and ask to commit**

```bash
git add sbijax/_src/abc/sabc.py sbijax/_src/abc/sabc_test.py
# then STOP — ask the user before committing.
# suggested message: "feat(abc): add SABC config and distance primitives"
```

---

## Task 2: Root finder and epsilon solvers

**Files:**
- Modify: `sbijax/_src/abc/sabc.py`
- Test: `sbijax/_src/abc/sabc_test.py`

- [ ] **Step 1: Write the failing test**

Append to `sbijax/_src/abc/sabc_test.py`:

```python
from scipy.optimize import brentq

from sbijax._src.abc.sabc import _epsilon_multi, _epsilon_single


def _ref_epsilon_single(ubar, v):
    if ubar <= 1e-12:
        return 0.0
    return brentq(lambda e: e**2 + v * e**1.5 - ubar**2, 0.0, ubar)


def _ref_epsilon_multi(u, v):
    # numpy port of epsilon.cpp::epsilon_multi
    u_bar = np.maximum(u.mean(axis=0), 1e-12)
    n = u.shape[1]
    cn = 1.0
    for k in range(1, n + 2):
        cn *= (n + 1 + k) / k
    cn /= n + 2
    out = np.zeros(n)
    for i in range(n):
        ub = min(u_bar[i], 0.5 - 1e-9)

        def g(beta):
            if beta < 1e-6:
                val = 0.5 - beta / 12.0
            else:
                e = np.exp(-beta)
                val = (1.0 - e * (1.0 + beta)) / (beta * (1.0 - e))
            return val - ub

        bhi = max(1.0, 10.0 / ub)
        while g(1e-6) * g(bhi) > 0.0:
            bhi *= 2.0
        beta = brentq(g, 1e-6, bhi)
        num = 1.0 + np.sum((u_bar / ub) ** (n / 2.0))
        prod = np.prod(u_bar / ub)
        den = cn * (n + 1) * ub ** (1.0 + n / 2.0) * prod
        out[i] = 1.0 / (beta + v * num / den)
    return out


def test_epsilon_single_matches_brentq():
    for ubar in [0.05, 0.2, 0.5, 0.9]:
        got = float(_epsilon_single(jnp.array(ubar), 1.0))
        np.testing.assert_allclose(got, _ref_epsilon_single(ubar, 1.0), rtol=1e-4)


def test_epsilon_single_zero_for_tiny_ubar():
    assert float(_epsilon_single(jnp.array(1e-15), 1.0)) == 0.0


def test_epsilon_multi_matches_reference():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.05, 0.45, size=(64, 3)).astype(np.float32)
    got = np.asarray(_epsilon_multi(jnp.asarray(u), 1.0))
    ref = _ref_epsilon_multi(u, 1.0)
    np.testing.assert_allclose(got, ref, rtol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k epsilon -v`
Expected: FAIL with `ImportError` (`_epsilon_single` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `sbijax/_src/abc/sabc.py` (after the distance callables):

```python
def _bisect(f, lo, hi, n_iter=60):
  """Bisection root finder for an increasing ``f`` with ``f(lo)<=0<=f(hi)``.

  Vectorized and jittable; ``lo``/``hi`` may be arrays (supports ``vmap``).
  """

  def body(_, bounds):
    lo, hi = bounds
    mid = 0.5 * (lo + hi)
    pos = f(mid) > 0
    return (jnp.where(pos, lo, mid), jnp.where(pos, mid, hi))

  lo, hi = lax.fori_loop(0, n_iter, body, (lo, hi))
  return 0.5 * (lo + hi)


def _epsilon_single(ubar, v):
  """Solve ``eps^2 + v*eps^1.5 - ubar^2 = 0`` on ``[0, ubar]`` (scalar)."""

  def f(eps):
    return eps**2 + v * eps**1.5 - ubar**2

  eps = _bisect(f, jnp.zeros_like(ubar), ubar)
  return jnp.where(ubar <= 1e-12, jnp.zeros_like(ubar), eps)


def _epsilon_multi(u, v):
  """Per-statistic epsilon via the reference root-find, shape ``(n_stats,)``."""
  n = u.shape[1]
  u_bar = jnp.maximum(jnp.mean(u, axis=0), 1e-12)

  cn = 1.0
  for k in range(1, n + 2):
    cn *= (n + 1 + k) / k
  cn /= n + 2

  def _val(beta):
    small = beta < 1e-6
    beta_safe = jnp.where(small, 1.0, beta)
    e = jnp.exp(-beta_safe)
    big = (1.0 - e * (1.0 + beta_safe)) / (beta_safe * (1.0 - e))
    return jnp.where(small, 0.5 - beta / 12.0, big)

  def _eps_one(ub):
    ub = jnp.minimum(ub, 0.5 - 1e-9)
    f = lambda beta: ub - _val(beta)  # noqa: E731 (increasing in beta)
    bhi = jnp.maximum(1.0, 10.0 / ub)
    bhi = lax.fori_loop(
      0, 60, lambda _, b: jnp.where(f(b) < 0, b * 2.0, b), bhi
    )
    beta = _bisect(f, jnp.asarray(1e-6), bhi)
    num = 1.0 + jnp.sum((u_bar / ub) ** (n / 2.0))
    prod = jnp.prod(u_bar / ub)
    den = cn * (n + 1) * ub ** (1.0 + n / 2.0) * prod
    return 1.0 / (beta + v * num / den)

  return jax.vmap(_eps_one)(u_bar)


def _epsilon(u, v, is_multi):
  """Dispatch to multi (``(n_stats,)``) or single (``(1,)``) epsilon."""
  if is_multi:
    return _epsilon_multi(u, v)
  return jnp.reshape(_epsilon_single(jnp.mean(u), v), (1,))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k epsilon -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Stage and ask to commit**

```bash
git add sbijax/_src/abc/sabc.py sbijax/_src/abc/sabc_test.py
# STOP — ask user. suggested: "feat(abc): add SABC epsilon root-finders"
```

---

## Task 3: Empirical-CDF transform

**Files:**
- Modify: `sbijax/_src/abc/sabc.py`
- Test: `sbijax/_src/abc/sabc_test.py`

- [ ] **Step 1: Write the failing test**

Append to `sbijax/_src/abc/sabc_test.py`:

```python
from sbijax._src.abc.sabc import _build_cdf, _cdf_eval


def test_cdf_eval_monotone_and_bounded():
    rng = np.random.default_rng(1)
    rho = jnp.asarray(rng.uniform(0.0, 5.0, size=(128, 2)).astype(np.float32))
    tables = _build_cdf(rho)
    u = _cdf_eval(tables, rho)
    assert u.shape == (128, 2)
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0
    # larger distances map to larger CDF values within a column
    for j in range(2):
        order = jnp.argsort(rho[:, j])
        col = u[order, j]
        assert bool(jnp.all(jnp.diff(col) >= -1e-5))


def test_cdf_eval_zero_distance_maps_to_zero():
    rho = jnp.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=jnp.float32)
    tables = _build_cdf(rho)
    u = _cdf_eval(tables, rho)
    np.testing.assert_allclose(np.asarray(u[0]), [0.0, 0.0], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k cdf -v`
Expected: FAIL with `ImportError` (`_build_cdf` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `sbijax/_src/abc/sabc.py`:

```python
def _build_cdf(rho):
  """Build per-statistic CDF knot tables from distances ``(B, n_stats)``.

  Each column is sorted, prepended with ``0`` and appended with
  ``1.5 * max``, paired with a uniform probability grid. Ties/zeros are not
  de-duplicated (matches the reference; SABC distances are continuous).

  Returns:
      ``(values, probs)`` each shape ``(B + 2, n_stats)``.
  """
  b, n = rho.shape
  ordered = jnp.sort(rho, axis=0)
  zeros = jnp.zeros((1, n), dtype=ordered.dtype)
  maxv = ordered[-1:, :] * _CDF_INFLATE
  values = jnp.concatenate([zeros, ordered, maxv], axis=0)
  probs = jnp.linspace(0.0, 1.0, b + 2, dtype=ordered.dtype)
  probs = jnp.broadcast_to(probs[:, None], (b + 2, n))
  return values, probs


def _cdf_eval(tables, rho):
  """Map distances ``(B, n_stats)`` to CDF probabilities via interpolation."""
  values, probs = tables

  def _interp(xs, ys, q):
    idx = jnp.sum(xs[None, :] <= q[:, None], axis=1)
    idx = jnp.clip(idx, 1, xs.shape[0] - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    t = jnp.clip((q - x0) / (x1 - x0), 0.0, 1.0)
    return y0 + t * (y1 - y0)

  return jax.vmap(_interp, in_axes=(1, 1, 1), out_axes=1)(values, probs, rho)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k cdf -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Stage and ask to commit**

```bash
git add sbijax/_src/abc/sabc.py sbijax/_src/abc/sabc_test.py
# STOP — ask user. suggested: "feat(abc): add SABC empirical-CDF transform"
```

---

## Task 4: Importance resampling

**Files:**
- Modify: `sbijax/_src/abc/sabc.py`
- Test: `sbijax/_src/abc/sabc_test.py`

- [ ] **Step 1: Write the failing test**

Append to `sbijax/_src/abc/sabc_test.py`:

```python
from sbijax._src.abc.sabc import _resample_indices, _resample_weights


def test_resample_weights_formula():
    u = jnp.array([[0.1, 0.1], [0.5, 0.5]])
    delta = 0.1
    u_bar = jnp.maximum(u.mean(axis=0, keepdims=True), 1e-12)
    expected = jnp.exp(-jnp.sum(u * (delta / u_bar), axis=1))
    np.testing.assert_allclose(_resample_weights(u, delta), expected, rtol=1e-6)


def test_resample_favours_small_distances():
    # particle 0 has tiny distances, particle 1 huge -> index 0 dominates
    u = jnp.array([[0.01, 0.01], [0.9, 0.9]])
    idx = _resample_indices(u, 0.1, 1000, jr.PRNGKey(0))
    assert idx.shape == (1000,)
    assert float(jnp.mean(idx == 0)) > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k resample -v`
Expected: FAIL with `ImportError` (`_resample_weights` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `sbijax/_src/abc/sabc.py`:

```python
def _resample_weights(u, delta):
  """Importance weights ``exp(-sum(delta * u / mean_u))`` over particles."""
  u_bar = jnp.maximum(jnp.mean(u, axis=0, keepdims=True), 1e-12)
  return jnp.exp(-jnp.sum(u * (delta / u_bar), axis=1))


def _resample_indices(u, delta, size, key):
  """Draw ``size`` resampling indices ~ Categorical(weights)."""
  logits = jnp.log(_resample_weights(u, delta))
  return jr.categorical(key, logits, shape=(size,))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k resample -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Stage and ask to commit**

```bash
git add sbijax/_src/abc/sabc.py sbijax/_src/abc/sabc_test.py
# STOP — ask user. suggested: "feat(abc): add SABC importance resampling"
```

---

## Task 5: Differential-Evolution proposal

**Files:**
- Modify: `sbijax/_src/abc/sabc.py`
- Test: `sbijax/_src/abc/sabc_test.py`

- [ ] **Step 1: Write the failing test**

Append to `sbijax/_src/abc/sabc_test.py`:

```python
from sbijax._src.abc.sabc import _de_propose


def test_de_propose_shape_and_zero_jitter():
    theta = jnp.zeros((4, 2))
    donors = jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    out = _de_propose(theta, donors, gamma0=0.5, sigma_gamma=0.0, key=jr.PRNGKey(0))
    assert out.shape == (4, 2)
    # with sigma_gamma=0, proposal = theta + 0.5 * (p1 - p2); difference is a
    # scaled difference of two donor rows, so each row lies on the donor lattice
    diffs = (out - theta) / 0.5
    # every diff must be a (p1 - p2) for some distinct donor pair
    donor_np = np.asarray(donors)
    for d in np.asarray(diffs):
        ok = any(
            np.allclose(d, donor_np[a] - donor_np[b])
            for a in range(4)
            for b in range(4)
            if a != b
        )
        assert ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k de_propose -v`
Expected: FAIL with `ImportError` (`_de_propose` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `sbijax/_src/abc/sabc.py`:

```python
def _de_propose(theta, donors, gamma0, sigma_gamma, key):
  """Differential-Evolution proposal ``theta + gamma * (p1 - p2)``.

  Partners ``p1, p2`` are distinct rows drawn uniformly from ``donors``.
  """
  b = theta.shape[0]
  m = donors.shape[0]
  k1, k2, k3 = jr.split(key, 3)
  i1 = jr.randint(k1, (b,), 0, m)
  i2 = jr.randint(k2, (b,), 0, m - 1)
  i2 = i2 + (i2 >= i1).astype(i2.dtype)
  p1, p2 = donors[i1], donors[i2]
  noise = jr.normal(k3, (b, 1))
  gamma = gamma0 * (1.0 + sigma_gamma * noise)
  return theta + gamma * (p1 - p2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k de_propose -v`
Expected: PASS (1 test).

- [ ] **Step 5: Stage and ask to commit**

```bash
git add sbijax/_src/abc/sabc.py sbijax/_src/abc/sabc_test.py
# STOP — ask user. suggested: "feat(abc): add SABC differential-evolution proposal"
```

---

## Task 6: Annealing core (`_update_half`, `_sabc_core`)

**Files:**
- Modify: `sbijax/_src/abc/sabc.py`
- Test: `sbijax/_src/abc/sabc_test.py`

- [ ] **Step 1: Write the failing test**

Append to `sbijax/_src/abc/sabc_test.py`:

```python
from sbijax._src.abc.sabc import _sabc_core


def _toy_problem():
    # 2-D Gaussian: theta ~ N(0, 3), y = theta + 0.1 * eps, observe [1, -1].
    n_para = 2
    ss_obs = jnp.array([1.0, -1.0])

    def rvs(key, size):
        return jr.normal(key, (size, n_para)) * 3.0

    def logpdf(theta):
        return jnp.sum(-0.5 * (theta / 3.0) ** 2 - jnp.log(3.0), axis=1)

    def simulate_distance(key, theta):
        y = theta + 0.1 * jr.normal(key, theta.shape)
        return jnp.abs(y - ss_obs)

    return rvs, logpdf, simulate_distance, n_para


def test_sabc_core_shapes_and_recovery():
    rvs, logpdf, sim, n_para = _toy_problem()
    out = _sabc_core(
        jr.PRNGKey(0), rvs, logpdf, sim,
        n_particles=2000, n_simulation=200_000,
        v=1.0, is_multi=False, gamma0=None, sigma_gamma=1e-5, delta=0.1,
    )
    population, u, rho, eps_hist, u_hist = out
    n_updates = 200_000 // 2000
    assert population.shape == (2000, n_para)
    assert eps_hist.shape[0] == n_updates + 1
    assert u_hist.shape == (n_updates + 1, 2)
    np.testing.assert_allclose(
        np.asarray(population.mean(axis=0)), [1.0, -1.0], atol=0.2
    )


def test_sabc_core_is_jittable():
    rvs, logpdf, sim, _ = _toy_problem()
    fn = jax.jit(
        _sabc_core,
        static_argnums=(1, 2, 3, 4, 5, 7, 8),
    )
    out = fn(
        jr.PRNGKey(1), rvs, logpdf, sim, 500, 5000, 1.0, False, None, 1e-5, 0.1
    )
    assert out[0].shape == (500, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k sabc_core -v`
Expected: FAIL with `ImportError` (`_sabc_core` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `sbijax/_src/abc/sabc.py`:

```python
def _update_half(
  state, lo, hi, donors, sim, logpdf, tables, inv_eps, gamma0, sigma_gamma, key
):
  """DE-MCMC update of the population slice ``[lo, hi)`` against ``donors``."""
  population, u, rho, logprior = state
  theta_cur = population[lo:hi]
  u_cur, rho_cur, lp_cur = u[lo:hi], rho[lo:hi], logprior[lo:hi]

  k_prop, k_acc, k_sim = jr.split(key, 3)
  theta_prop = _de_propose(theta_cur, donors, gamma0, sigma_gamma, k_prop)
  lp_prop = logpdf(theta_prop)
  rho_prop = sim(k_sim, theta_prop)
  u_prop = _cdf_eval(tables, rho_prop)

  dterm = jnp.sum((u_cur - u_prop) * inv_eps, axis=1)
  log_acc = lp_prop - lp_cur + dterm
  finite = jnp.isfinite(lp_prop)
  u_unif = jr.uniform(k_acc, (hi - lo,))
  accept = finite & (jnp.log(u_unif) < log_acc)
  col = accept[:, None]

  population = population.at[lo:hi].set(jnp.where(col, theta_prop, theta_cur))
  u = u.at[lo:hi].set(jnp.where(col, u_prop, u_cur))
  rho = rho.at[lo:hi].set(jnp.where(col, rho_prop, rho_cur))
  logprior = logprior.at[lo:hi].set(jnp.where(accept, lp_prop, lp_cur))
  return population, u, rho, logprior


def _sabc_core(
  key, rvs, logpdf, sim, n_particles, n_simulation, v, is_multi,
  gamma0, sigma_gamma, delta,
):
  """Run the SABC annealing loop on raveled arrays.

  Args:
      key: PRNG key.
      rvs: ``(key, size) -> (N, n_para)`` prior sampler.
      logpdf: ``(N, n_para) -> (N,)`` prior log-density.
      sim: ``(key, (B, n_para)) -> (B, n_stats)`` simulate-and-distance fn.
      n_particles: population size ``N``.
      n_simulation: total simulation budget.
      v: annealing speed.
      is_multi: ``True`` for per-statistic epsilon, else a single shared eps.
      gamma0: DE step; ``None`` -> ``2.38/sqrt(2*n_para)``.
      sigma_gamma: DE jitter.
      delta: resampling temperature.

  Returns:
      ``(population, u, rho, epsilon_history, u_history)``.
  """
  k0, k_sim0, k_rs, key = jr.split(key, 4)
  population = rvs(k0, n_particles)
  n_para = population.shape[1]
  rho = sim(k_sim0, population)
  logprior = logpdf(population)
  tables = _build_cdf(rho)
  u = _cdf_eval(tables, rho)

  idx = _resample_indices(u, delta, n_particles, k_rs)
  population, u, rho, logprior = (
    population[idx], u[idx], rho[idx], logprior[idx]
  )

  if gamma0 is None or gamma0 <= 0:
    gamma0 = 2.38 / (2.0 * n_para) ** 0.5

  epsilon = _epsilon(u, v, is_multi)
  mid = n_particles // 2
  n_updates = n_simulation // n_particles

  def step(carry, it):
    population, u, rho, logprior, epsilon, key = carry
    inv_eps = 1.0 / epsilon
    k1, k2, k_rs, key = jr.split(key, 4)
    state = (population, u, rho, logprior)
    state = _update_half(
      state, 0, mid, population[mid:], sim, logpdf, tables,
      inv_eps, gamma0, sigma_gamma, k1,
    )
    state = _update_half(
      state, mid, n_particles, state[0][:mid], sim, logpdf, tables,
      inv_eps, gamma0, sigma_gamma, k2,
    )
    # reference resampling cadence: every 2 iterations starting at it == 3.
    do_rs = jnp.logical_and(jnp.mod(it, 2) == 1, it >= 3)
    ridx = _resample_indices(state[1], delta, n_particles, k_rs)
    state = lax.cond(
      do_rs,
      lambda s: tuple(x[ridx] for x in s),
      lambda s: s,
      state,
    )
    population, u, rho, logprior = state
    epsilon = _epsilon(u, v, is_multi)
    carry = (population, u, rho, logprior, epsilon, key)
    return carry, (epsilon, jnp.mean(u, axis=0))

  init = (population, u, rho, logprior, epsilon, key)
  final, (eps_hist, u_hist) = lax.scan(init=init, xs=jnp.arange(n_updates), f=step)
  population, u, rho = final[0], final[1], final[2]
  eps_history = jnp.concatenate([epsilon[None], eps_hist], axis=0)
  u_history = jnp.concatenate([jnp.mean(init[1], axis=0)[None], u_hist], axis=0)
  return population, u, rho, eps_history, u_history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k sabc_core -v`
Expected: PASS (2 tests). The recovery test may take a few seconds.

- [ ] **Step 5: Stage and ask to commit**

```bash
git add sbijax/_src/abc/sabc.py sbijax/_src/abc/sabc_test.py
# STOP — ask user. suggested: "feat(abc): add SABC annealing core loop"
```

---

## Task 7: `SABC` class and `sample_posterior`

**Files:**
- Modify: `sbijax/_src/abc/sabc.py`
- Test: `sbijax/_src/abc/sabc_test.py`

- [ ] **Step 1: Write the failing test**

Append to `sbijax/_src/abc/sabc_test.py`:

```python
from sbijax._src.abc.sabc import SABC


def test_sabc_recovers_gaussian_single_eps(prior_simulator_tuple):
    y_observed = jnp.array([1.0, -1.0])
    model = SABC(prior_simulator_tuple, summary_fn=lambda x: x)
    idata, info = model.sample_posterior(
        jr.PRNGKey(0), y_observed,
        n_particles=2000, n_simulation=200_000, schedule=SingleEps(v=1.0),
    )
    samples = np.asarray(idata.posterior["theta"]).reshape(-1, 2)
    # conftest prior N(0,1), likelihood N(theta,1) -> posterior mean obs/2.
    np.testing.assert_allclose(samples.mean(axis=0), [0.5, -0.5], atol=0.2)
    n_updates = 200_000 // 2000
    assert len(info.epsilon_history) == n_updates + 1


def test_sabc_recovers_gaussian_multi_eps(prior_simulator_tuple):
    y_observed = jnp.array([1.0, -1.0])
    model = SABC(prior_simulator_tuple, summary_fn=lambda x: x)
    idata, _ = model.sample_posterior(
        jr.PRNGKey(0), y_observed,
        n_particles=2000, n_simulation=200_000, schedule=MultiEps(v=1.0),
    )
    samples = np.asarray(idata.posterior["theta"]).reshape(-1, 2)
    np.testing.assert_allclose(samples.mean(axis=0), [0.5, -0.5], atol=0.2)
```

Note: the `prior_simulator_tuple` fixture lives in `sbijax/_src/conftest.py`
(2-D Gaussian, `theta ~ N(0, 1)`, `y = theta + N(0, 1)`). The observed
`[1, -1]` yields a posterior mean near `[0.5, -0.5]` (Gaussian conjugate:
`obs/2`); the `atol=0.2` band covers it. Confirm the fixture's prior scale
before tightening tolerances.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k recovers -v`
Expected: FAIL with `ImportError` (`SABC` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `sbijax/_src/abc/sabc.py`:

```python
sabc_info = namedtuple("sabc_info", "epsilon_history u_history rho")


class SABC(SBI):
  r"""Simulated Annealing approximate Bayesian computation.

  Implements the algorithm from :cite:t:`albert2025simulated`. Unlike
  :class:`~sbijax.SMCABC`, the ``distance_fn`` must return a per-statistic
  array of shape ``(n_particles, n_stats)`` (it must not collapse to a scalar):
  the multi-epsilon schedule assigns one temperature per summary statistic.

  Args:
      model_fns: tuple ``(prior_fn, simulator_fn)``; ``prior_fn`` builds a
          ``tfd.JointDistributionNamed`` and ``simulator_fn(seed, theta)``
          simulates data.
      summary_fn: maps simulated data to summary statistics.
      distance_fn: ``(summary_simulated, summary_observed) -> (B, n_stats)``.

  Examples:
      >>> from sbijax import SABC
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      >>> prior = lambda: tfd.JointDistributionNamed(
      ...     dict(theta=tfd.Normal(jnp.zeros(2), 1.0)), batch_ndims=0)
      >>> sim = lambda seed, theta: theta["theta"] + tfd.Normal(
      ...     0.0, 0.1).sample(theta["theta"].shape, seed=seed)
      >>> model = SABC((prior, sim), lambda x: x)

  References:
      Albert, Carlo, et al. "Simulated Annealing ABC with multiple summary
      statistics". arXiv preprint arXiv:2505.23261, 2025.
  """

  def __init__(self, model_fns, summary_fn=lambda x: x, distance_fn=abs_distance):
    super().__init__(model_fns)
    self.summary_fn = summary_fn
    self.distance_fn = distance_fn

  def sample_posterior(
    self,
    rng_key,
    observable,
    n_particles=1000,
    n_simulation=100_000,
    schedule=None,
    proposal=None,
    delta=0.1,
  ):
    r"""Sample from the SABC approximate posterior.

    Args:
        rng_key: a JAX PRNG key.
        observable: the observation to condition on.
        n_particles: population size.
        n_simulation: total simulation budget.
        schedule: ``SingleEps`` (default) or ``MultiEps``.
        proposal: ``DiffEvolution`` (default).
        delta: resampling temperature (positive).

    Returns:
        a tuple ``(InferenceData, sabc_info)``.
    """
    schedule = schedule or SingleEps()
    proposal = proposal or DiffEvolution()
    is_multi = isinstance(schedule, MultiEps)

    probe = self.prior.sample(seed=jr.PRNGKey(0))
    _, unravel = ravel_pytree(probe)

    def rvs(key, size):
      sample = self.prior.sample(seed=key, sample_shape=(size,))
      return jax.vmap(lambda x: ravel_pytree(x)[0])(sample)

    def logpdf(theta_flat):
      return self.prior.log_prob(jax.vmap(unravel)(theta_flat))

    ss_obs = self.summary_fn(observable)

    def sim(key, theta_flat):
      theta = jax.vmap(unravel)(theta_flat)
      y = self.simulator_fn(seed=key, theta=theta)
      return self.distance_fn(self.summary_fn(y), ss_obs)

    population, u, rho, eps_hist, u_hist = _sabc_core(
      rng_key, rvs, logpdf, sim,
      n_particles, n_simulation,
      float(schedule.v), is_multi,
      proposal.gamma0, float(proposal.sigma_gamma), float(delta),
    )

    named = jax.vmap(unravel)(population)
    thetas = jax.tree_util.tree_map(lambda x: x.reshape(1, *x.shape), named)
    idata = as_inference_data(thetas, jnp.squeeze(observable))
    info = sabc_info(
      epsilon_history=eps_hist, u_history=u_hist, rho=rho
    )
    return idata, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k recovers -v`
Expected: PASS (2 tests). Each runs the full sampler; allow ~10-20s.

- [ ] **Step 5: Run the full SABC test module**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -v`
Expected: PASS (all tests from Tasks 1-7).

- [ ] **Step 6: Stage and ask to commit**

```bash
git add sbijax/_src/abc/sabc.py sbijax/_src/abc/sabc_test.py
# STOP — ask user. suggested: "feat(abc): add SABC class and sample_posterior"
```

---

## Task 8: Public exports

**Files:**
- Modify: `sbijax/__init__.py`
- Test: `sbijax/_src/abc/sabc_test.py`

- [ ] **Step 1: Write the failing test**

Append to `sbijax/_src/abc/sabc_test.py`:

```python
def test_public_exports():
    import sbijax

    assert hasattr(sbijax, "SABC")
    assert hasattr(sbijax, "SingleEps")
    assert hasattr(sbijax, "MultiEps")
    assert hasattr(sbijax, "DiffEvolution")
    assert hasattr(sbijax, "abs_distance")
    assert hasattr(sbijax, "sq_distance")
    assert hasattr(sbijax, "weighted_sq")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k public_exports -v`
Expected: FAIL with `AssertionError` (`sbijax` has no attribute `SABC`).

- [ ] **Step 3: Write minimal implementation**

In `sbijax/__init__.py`, add the import after the existing
`from sbijax._src.abc.smc_abc import SMCABC` line:

```python
from sbijax._src.abc.sabc import (
  SABC,
  DiffEvolution,
  MultiEps,
  SingleEps,
  abs_distance,
  sq_distance,
  weighted_sq,
)
```

And extend the `__all__` list (add these entries):

```python
  "SABC",
  "SingleEps",
  "MultiEps",
  "DiffEvolution",
  "abs_distance",
  "sq_distance",
  "weighted_sq",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest sbijax/_src/abc/sabc_test.py -k public_exports -v`
Expected: PASS (1 test).

- [ ] **Step 5: Stage and ask to commit**

```bash
git add sbijax/__init__.py sbijax/_src/abc/sabc_test.py
# STOP — ask user. suggested: "feat(abc): export SABC public API"
```

---

## Task 9: Documentation

**Files:**
- Modify: `docs/references.bib`, `docs/sbijax.rst`

- [ ] **Step 1: Add the citation**

Append to `docs/references.bib`:

```bibtex
@article{albert2025simulated,
  title={Simulated Annealing ABC with multiple summary statistics},
  author={Albert, Carlo and Ulzega, Simone and Dirmeier, Simon and Scheidegger, Andreas and Bassi, Alberto and Mira, Antonietta},
  journal={arXiv preprint arXiv:2505.23261},
  year={2025}
}
```

- [ ] **Step 2: Document `SABC` in the API reference**

Inspect `docs/sbijax.rst` to match the existing layout, then add `SABC`
alongside `SMCABC` (an `autoclass`/autosummary entry mirroring how `SMCABC`
is documented). Run:

`uv run python -c "import sbijax; help(sbijax.SABC)"`
Expected: the docstring renders without error.

- [ ] **Step 3: Verify the full suite and lints**

Run: `uv run pytest sbijax/_src/abc/ -v`
Expected: PASS (all ABC tests).

Run: `make lints && make format`
Expected: no errors (auto-fixes applied).

- [ ] **Step 4: Stage and ask to commit**

```bash
git add docs/references.bib docs/sbijax.rst
# STOP — ask user. suggested: "docs(abc): document SABC method"
```

---

## Self-review notes

- **Spec coverage:** SingleEps/MultiEps (Task 2, 6, 7), DE-MCMC (Task 5, 6),
  CDF transform (Task 3), resampling (Task 4), all distances via callables
  (Task 1), histories (Task 6, 7), InferenceData return (Task 7), exports
  (Task 8), docs+citation (Task 9). The spec's three reference distances map to
  `abs_distance`/`sq_distance`/`weighted_sq`.
- **Type consistency:** `_sabc_core` returns
  `(population, u, rho, epsilon_history, u_history)` everywhere it is called;
  `sabc_info` carries `epsilon_history, u_history, rho`. `_epsilon` returns
  shape `(1,)` (single) or `(n_stats,)` (multi), consistent with the
  `inv_eps` broadcast in `_update_half`.
- **Known divergences from the reference (intentional):** explicit PRNG keys
  thread through the simulator (MLX used a global RNG); the resample cadence is
  the closed-form equivalent (`it odd and it >= 3`) of the reference's proxy
  counter; Brent is replaced by fixed-iteration bisection.
- **Risk:** the recovery tolerances (`atol`) assume the conftest prior scale;
  Task 7 Step 1 flags confirming the fixture before tightening.
