r"""Simulated Annealing ABC (SABC).

JAX port of the ``sabc`` reference implementation. Implements the algorithm
from :cite:t:`albert2025simulated`.

References:
    Albert, Carlo, et al. "Simulated Annealing ABC with multiple summary
    statistics". arXiv preprint arXiv:2505.23261, 2025.
"""

# ruff: noqa: PLR0913
from collections import namedtuple
from dataclasses import dataclass
from functools import partial

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
  """Absolute per-dimension distance ``|simulated - observed|``, ``(B, n)``."""
  return jnp.abs(simulated - observed)


def sq_distance(simulated, observed):
  """Squared per-dimension distance ``(simulated - observed)**2``."""
  return jnp.square(simulated - observed)


def l2_distance(simulated, observed):
  """Scalar Euclidean distance over the last axis, shape ``(B, 1)``.

  Use this for a single aggregated (scalar) distance; pair it with either
  ``SingleEps`` or ``MultiEps`` (the latter degenerates to a single epsilon
  when there is one statistic).
  """
  d = jnp.sqrt(
    jnp.sum(jnp.square(simulated - observed), axis=-1, keepdims=True)
  )
  return d


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
    idx = jnp.searchsorted(xs, q, side="right")
    idx = jnp.clip(idx, 1, xs.shape[0] - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    t = jnp.clip((q - x0) / (x1 - x0), 0.0, 1.0)
    return y0 + t * (y1 - y0)

  return jax.vmap(_interp, in_axes=(1, 1, 1), out_axes=1)(values, probs, rho)


def _resample_weights(u, delta):
  """Importance weights ``exp(-sum(delta * u / mean_u))`` over particles."""
  u_bar = jnp.maximum(jnp.mean(u, axis=0, keepdims=True), 1e-12)
  return jnp.exp(-jnp.sum(u * (delta / u_bar), axis=1))


def _resample_indices(u, delta, size, key):
  """Draw ``size`` resampling indices ~ Categorical(weights).

  Inverse-CDF sampling (cumsum + searchsorted), O(N log N). Avoids
  ``jr.categorical``, which is Gumbel-max: it materializes a
  ``(size, n_categories)`` array (O(N^2) time and memory when ``size == N``).
  """
  w = _resample_weights(u, delta)
  cdf = jnp.cumsum(w)
  unif = jr.uniform(key, (size,)) * cdf[-1]
  idx = jnp.searchsorted(cdf, unif, side="right")
  return jnp.minimum(idx, size - 1)


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
  return (population, u, rho, logprior), jnp.sum(accept)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 7, 8))
def _sabc_core(
  key,
  rvs,
  logpdf,
  sim,
  n_particles,
  n_simulation,
  v,
  is_multi,
  gamma0,
  sigma_gamma,
  delta,
):
  """Run the SABC annealing loop on raveled arrays.

  JIT-compiled so the whole annealing loop runs as one fused executable;
  ``rvs``/``logpdf``/``sim`` and the sizes/flags are static.

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
    population[idx],
    u[idx],
    rho[idx],
    logprior[idx],
  )

  if gamma0 is None or gamma0 <= 0:
    gamma0 = 2.38 / (2.0 * n_para) ** 0.5

  epsilon = _epsilon(u, v, is_multi)
  mid = n_particles // 2
  n_updates = n_simulation // n_particles
  resample_interval = 2 * n_particles

  def step(carry, _it):
    population, u, rho, logprior, epsilon, n_acc, n_rs, key = carry
    inv_eps = 1.0 / epsilon
    k1, k2, k_rs, key = jr.split(key, 4)
    state = (population, u, rho, logprior)
    state, a1 = _update_half(
      state,
      0,
      mid,
      population[mid:],
      sim,
      logpdf,
      tables,
      inv_eps,
      gamma0,
      sigma_gamma,
      k1,
    )
    state, a2 = _update_half(
      state,
      mid,
      n_particles,
      state[0][:mid],
      sim,
      logpdf,
      tables,
      inv_eps,
      gamma0,
      sigma_gamma,
      k2,
    )
    # Resample each time the cumulative number of accepted proposals crosses
    # the next ``2 * n_particles`` threshold (the reference SABC cadence).
    n_acc = (n_acc + a1 + a2).astype(n_acc.dtype)
    do_rs = n_acc >= (n_rs + 1) * resample_interval
    ridx = _resample_indices(state[1], delta, n_particles, k_rs)
    state = lax.cond(
      do_rs,
      lambda s: tuple(x[ridx] for x in s),
      lambda s: s,
      state,
    )
    n_rs = n_rs + do_rs.astype(n_rs.dtype)
    population, u, rho, logprior = state
    epsilon = _epsilon(u, v, is_multi)
    carry = (population, u, rho, logprior, epsilon, n_acc, n_rs, key)
    return carry, (epsilon, jnp.mean(u, axis=0))

  init = (
    population,
    u,
    rho,
    logprior,
    epsilon,
    jnp.zeros((), jnp.int32),
    jnp.zeros((), jnp.int32),
    key,
  )
  final, (eps_hist, u_hist) = lax.scan(
    f=step, init=init, xs=jnp.arange(n_updates)
  )
  population, u, rho = final[0], final[1], final[2]
  eps_history = jnp.concatenate([epsilon[None], eps_hist], axis=0)
  u_history = jnp.concatenate([jnp.mean(init[1], axis=0)[None], u_hist], axis=0)
  return population, u, rho, eps_history, u_history


sabc_info = namedtuple("sabc_info", "epsilon_history u_history rho")


class SABC(SBI):
  r"""Simulated Annealing approximate Bayesian computation.

  Implements the algorithm from :cite:t:`albert2025simulated`. The
  ``distance_fn`` may be either *scalar* (one aggregated distance per particle,
  e.g. :func:`l2_distance`) or *per-dimension* (one distance per summary
  statistic, e.g. :func:`abs_distance`). This is orthogonal to the epsilon
  schedule: :class:`SingleEps` uses one shared temperature, :class:`MultiEps`
  assigns one temperature per statistic (the paper's contribution, meaningful
  only with a per-dimension distance). All four combinations are valid.

  Args:
      model_fns: tuple ``(prior_fn, simulator_fn)``; ``prior_fn`` builds a
          ``tfd.JointDistributionNamed`` and ``simulator_fn(seed, theta)``
          simulates data.
      summary_fn: maps simulated data to summary statistics.
      distance_fn: ``(summary_simulated, summary_observed) -> (B, n_stats)``
          (per-dimension) or ``-> (B,)`` / ``(B, 1)`` (scalar); scalar outputs
          are reshaped to ``(B, 1)`` internally.

  Examples:
      >>> from sbijax import SABC
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      >>> prior = tfd.JointDistributionNamed(
      ...     dict(theta=tfd.Normal(jnp.zeros(2), 1.0)), batch_ndims=0)
      >>> sim = lambda seed, theta: theta["theta"] + tfd.Normal(
      ...     0.0, 0.1).sample(theta["theta"].shape, seed=seed)
      >>> model = SABC((prior, sim), lambda x: x)

  References:
      Albert, Carlo, et al. "Simulated Annealing ABC with multiple summary
      statistics". arXiv preprint arXiv:2505.23261, 2025.
  """

  def __init__(
    self, model_fns, summary_fn=lambda x: x, distance_fn=abs_distance
  ):
    super().__init__(model_fns)
    self.summary_fn = summary_fn
    self.distance_fn = distance_fn
    self._rvs = None
    self._logpdf = None
    self._unravel = None
    self._sim = None
    self._sim_obs_id = None

  def _build_fns(self, observable):
    r"""Build and cache the raveled prior/simulator closures.

    Reusing the same callable objects across calls lets the jitted
    ``_sabc_core`` hit its trace cache instead of re-tracing the whole scan
    every call. ``rvs``/``logpdf`` depend only on the prior; ``sim`` is
    rebuilt only when the observation changes.
    """
    if self._rvs is None:
      probe = self.prior.sample(seed=jr.PRNGKey(0))
      _, unravel = ravel_pytree(probe)
      self._unravel = unravel

      def rvs(key, size):
        sample = self.prior.sample(seed=key, sample_shape=(size,))
        return jax.vmap(lambda x: ravel_pytree(x)[0])(sample)

      def logpdf(theta_flat):
        return self.prior.log_prob(jax.vmap(unravel)(theta_flat))

      self._rvs, self._logpdf = rvs, logpdf

    if self._sim_obs_id != id(observable):
      unravel = self._unravel
      ss_obs = self.summary_fn(observable)

      def sim(key, theta_flat):
        theta = jax.vmap(unravel)(theta_flat)
        y = self.simulator_fn(seed=key, theta=theta)
        d = self.distance_fn(self.summary_fn(y), ss_obs)
        # scalar ((B,)/(B,1)) or per-dimension ((B, n_stats)) distances.
        return jnp.reshape(d, (theta_flat.shape[0], -1))

      self._sim, self._sim_obs_id = sim, id(observable)

    return self._rvs, self._logpdf, self._sim, self._unravel

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

    rvs, logpdf, sim, unravel = self._build_fns(observable)

    population, u, rho, eps_hist, u_hist = _sabc_core(
      rng_key,
      rvs,
      logpdf,
      sim,
      n_particles,
      n_simulation,
      float(schedule.v),
      is_multi,
      proposal.gamma0,
      float(proposal.sigma_gamma),
      float(delta),
    )

    named = jax.vmap(unravel)(population)
    thetas = jax.tree_util.tree_map(lambda x: x.reshape(1, *x.shape), named)
    idata = as_inference_data(thetas, jnp.squeeze(observable))
    info = sabc_info(epsilon_history=eps_hist, u_history=u_hist, rho=rho)
    return idata, info
