# pylint: skip-file

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random as jr
from scipy.optimize import brentq
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.abc.sabc import (
    SABC,
    DiffEvolution,
    MultiEps,
    SingleEps,
    _build_cdf,
    _cdf_eval,
    _de_propose,
    _epsilon_multi,
    _epsilon_single,
    _resample_indices,
    _resample_weights,
    _sabc_core,
    abs_distance,
    l2_distance,
    sq_distance,
    weighted_sq,
)

# --- Task 1: configs and distances -----------------------------------------


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


# --- Task 2: epsilon solvers -----------------------------------------------


def _ref_epsilon_single(ubar, v):
    if ubar <= 1e-12:
        return 0.0
    return brentq(lambda e: e**2 + v * e**1.5 - ubar**2, 0.0, ubar)


def _ref_epsilon_multi(u, v):
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


# --- Task 3: CDF transform -------------------------------------------------


def test_cdf_eval_monotone_and_bounded():
    rng = np.random.default_rng(1)
    rho = jnp.asarray(rng.uniform(0.0, 5.0, size=(128, 2)).astype(np.float32))
    tables = _build_cdf(rho)
    u = _cdf_eval(tables, rho)
    assert u.shape == (128, 2)
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0
    for j in range(2):
        order = jnp.argsort(rho[:, j])
        col = u[order, j]
        assert bool(jnp.all(jnp.diff(col) >= -1e-5))


def test_cdf_eval_smallest_distance_maps_to_column_min():
    # No-dedup knot tables (matches the C++ reference): the prepended 0 plus a
    # data zero shift the grid, so the smallest distance maps to the column
    # minimum rather than exactly 0. Validate that contract.
    rho = jnp.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=jnp.float32)
    tables = _build_cdf(rho)
    u = _cdf_eval(tables, rho)
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0
    np.testing.assert_allclose(np.asarray(u[0]), np.asarray(u.min(axis=0)))
    assert bool(jnp.all(u[0] < u[2]))


# --- Task 4: resampling ----------------------------------------------------


def test_resample_weights_formula():
    u = jnp.array([[0.1, 0.1], [0.5, 0.5]])
    delta = 0.1
    u_bar = jnp.maximum(u.mean(axis=0, keepdims=True), 1e-12)
    expected = jnp.exp(-jnp.sum(u * (delta / u_bar), axis=1))
    np.testing.assert_allclose(_resample_weights(u, delta), expected, rtol=1e-6)


def test_resample_favours_small_distances():
    # delta scales the discrimination sharpness; delta=1 strongly favours the
    # low-distance particle.
    u = jnp.array([[0.01, 0.01], [0.9, 0.9]])
    idx = _resample_indices(u, 1.0, 1000, jr.PRNGKey(0))
    assert idx.shape == (1000,)
    assert float(jnp.mean(idx == 0)) > 0.9


# --- Task 5: DE proposal ---------------------------------------------------


def test_de_propose_shape_and_zero_jitter():
    theta = jnp.zeros((4, 2))
    donors = jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    out = _de_propose(theta, donors, gamma0=0.5, sigma_gamma=0.0, key=jr.PRNGKey(0))
    assert out.shape == (4, 2)
    diffs = (out - theta) / 0.5
    donor_np = np.asarray(donors)
    for d in np.asarray(diffs):
        ok = any(
            np.allclose(d, donor_np[a] - donor_np[b])
            for a in range(4)
            for b in range(4)
            if a != b
        )
        assert ok


# --- Task 6: annealing core ------------------------------------------------


def _toy_problem():
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


# --- Task 7: SABC class (2x2: single/multi eps x scalar/vector distance) ----


def _low_noise_model():
    # wide prior, low simulator noise -> posterior concentrates near observed.
    def prior_fn():
        return tfd.JointDistributionNamed(
            dict(theta=tfd.Normal(jnp.zeros(2), 3.0)), batch_ndims=0
        )

    def simulator_fn(seed, theta):
        return theta["theta"] + tfd.Normal(0.0, 0.1).sample(
            theta["theta"].shape, seed=seed
        )

    return prior_fn, simulator_fn


@pytest.mark.parametrize(
    "schedule",
    [SingleEps(v=1.0), MultiEps(v=1.0)],
    ids=["single_eps", "multi_eps"],
)
@pytest.mark.parametrize(
    "distance_fn",
    [abs_distance, l2_distance],
    ids=["vector_dist", "scalar_dist"],
)
def test_sabc_recovers_low_noise_gaussian(schedule, distance_fn):
    y_observed = jnp.array([1.0, -1.0])
    model = SABC(_low_noise_model(), summary_fn=lambda x: x, distance_fn=distance_fn)
    idata, info = model.sample_posterior(
        jr.PRNGKey(0), y_observed,
        n_particles=2000, n_simulation=200_000, schedule=schedule,
    )
    samples = np.asarray(idata.posterior["theta"]).reshape(-1, 2)
    np.testing.assert_allclose(samples.mean(axis=0), [1.0, -1.0], atol=0.15)
    n_updates = 200_000 // 2000
    assert len(info.epsilon_history) == n_updates + 1


# --- Task 8: exports -------------------------------------------------------


def test_public_exports():
    import sbijax

    assert hasattr(sbijax, "SABC")
    assert hasattr(sbijax, "SingleEps")
    assert hasattr(sbijax, "MultiEps")
    assert hasattr(sbijax, "DiffEvolution")
    assert hasattr(sbijax, "abs_distance")
    assert hasattr(sbijax, "sq_distance")
    assert hasattr(sbijax, "l2_distance")
    assert hasattr(sbijax, "weighted_sq")
