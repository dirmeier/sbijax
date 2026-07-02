# sbijax 0.4 redesign — backlog

Remaining work after the functional core and the ten-method port. See
[architecture.md](architecture.md) and [decisions.md](decisions.md).

**All items below are implemented** (items 1–5). This document is retained as
the design record for that work.

## 1. Sequential inference driver (`run_sequential`) + atomic NPE

**Status:** done. `npe(prior, net, *, num_atoms=10)` selects the atomic
proposal-posterior loss when `fit` is passed an `Info` with `round > 0`;
`run_sequential` drives the rounds (see `inference/sequential.py`, DR-011).

### Proposed design

A standalone driver orchestrates the rounds; the estimator stays otherwise
stateless.

```
run_sequential(
    rng_key, estimator, prior, simulator, observable,
    *, n_rounds, n_simulations_per_round, **fit_kwargs
) -> (params, info)
```

Loop:

```
data, params, info = None, None, None
for _ in range(n_rounds):
    proposal = None if info is None else _posterior_proposal(estimator, params, observable)
    round_data = simulate(key, prior, simulator, proposal=proposal,
                          n=n_simulations_per_round)
    data = round_data if data is None else stack(data, round_data)
    params, info = estimator.fit(key, data, info=info, **fit_kwargs)
return params, info
```

Two seams make this work:

- **`estimator.fit(..., info=None)`** gains an optional `info` argument
  (DR-011). `info is None` is round 0; otherwise the round is `info.round + 1`.
  NLE/NRE ignore the round — their loss is proposal-invariant — and just pass it
  through. NPE reads it: round 0 uses the maximum-likelihood loss (with
  event-space bijections), round > 0 uses the atomic proposal-posterior loss.
  The atomic loss needs only the prior (already captured) and `num_atoms` (a
  factory argument), so no proposal density has to be threaded into the loss.
  `round` is the only field `fit` reads back; `Info` is otherwise output-only
  diagnostics (per-method `NPEInfo`/`NLEInfo`/..., DR-011).
- **`_posterior_proposal(estimator, params, observable)`** returns a callable
  `(rng_key, n) -> theta` that draws from the current posterior. For amortized
  posterior methods this wraps `estimator.sample`; for NLE/NRE it wraps their
  MCMC `sample`. It lives next to the driver, not on the estimator.

### Atomic NPE loss (round > 0)

When `round > 0`, NPE must correct for the proposal no longer being the prior.
The atomic (APT / NPE-C) loss does this contrastively within the batch, using
only the network and the prior — no proposal density. Port of the deleted
`NPE._proposal_posterior_log_prob`:

```
def _atomic_loss(network, prior, params, rng, num_atoms, theta, y):
    n = theta.shape[0]
    m = clip(num_atoms, 2, n)                       # atoms per contrast set
    # for each row, pick m-1 *other* rows uniformly (exclude self)
    probs = (1 - eye(n)) / (n - 1)
    contrast = sample_without_replacement(rng, probs, k=m-1)   # (n, m-1)
    atomic_theta = concat(theta[:, None], theta[contrast], axis=1)  # (n, m, d)
    atomic_theta = atomic_theta.reshape(n * m, d)
    y_rep = repeat(y, m, axis=0)                    # (n*m, .)
    lp_post = network.log_prob(y=atomic_theta, x=y_rep).reshape(n, m)
    lp_prior = prior.log_prob(atomic_theta).reshape(n, m)
    unnorm = lp_post - lp_prior                     # importance-reweight
    # the true theta sits at atom index 0 of each set
    log_prob = unnorm[:, 0] - logsumexp(unnorm, axis=-1)
    return -mean(log_prob)
```

`num_atoms` is a factory argument of `npe`; `round` selects between this and the
round-0 maximum-likelihood loss inside `npe`'s `fit`. Everything else about the
estimator is unchanged.

### Open questions

- ~~Whether `round` on `fit` is the cleanest signal.~~ *Resolved (DR-011):* the
  round is carried by the per-method `Info` (`fit(..., info=None)`), keeping one
  `fit` signature; `round` is the only field read back.
- Proposal handoff for MCMC-sampled methods (NLE/NRE): drawing `n` proposal
  parameters per round via MCMC may be expensive; consider caching or reusing
  chains.
- ~~SNPE truncation (NPSE/AiO) vs atomic correction — truncated proposals a
  separate driver option?~~ *Resolved:* truncation is a driver **option** —
  `run_sequential` takes a `proposal_fn` hook, and
  `experimental.make_truncated_proposal` builds a truncated-prior proposal for
  it (see item 3).

## 2. Public API cutover

**Status:** done. `sbijax/__init__.py` exports the factories (`nle`, `npe`,
`nre`, `fmpe`, `cmpe`, `snle`, `sabc`, `smcabc`, `nass`, `nasss`), `simulate`,
`stack`, `run_sequential`, `sbc`, the interface records (`Estimator`,
`ABCSampler`, `SummaryNet`) and the per-method `Info` records. Breaking 0.4
surface (DR-003, DR-010).

## 3. Experimental methods

**Status:** done. `experimental/npse.py` and `experimental/aio.py` are
functional factories delegating to the `fmpe` core; their truncated-prior
sampling is ported to `experimental/_truncated.py`'s `make_truncated_proposal`,
plugged into `run_sequential` via its `proposal_fn` hook.

## 4. Correctness harness — remaining tiers

**Status:** done (DR-009). Beyond the conformance tier:

- **SBC calibration** — `diagnostics/sbc.py` (`sbc`), with a calibration test on
  a tractable Gaussian problem (`diagnostics/sbc_test.py`).
- **Benchmark table** — `diagnostics/benchmark_test.py` checks mean/covariance
  recovery of the analytic Gaussian posterior for `npe` and `nle`.

## 5. Docs and examples

**Status:** done.

- Migration guide (`docs/migration.rst`), linked from the docs index.
- `examples/` rewritten to the functional API (`amortized_npe.py`,
  `sequential_npe.py`); front-page example updated.
- Sphinx reference (`docs/sbijax.rst`, `docs/sbijax.experimental.rst`) updated
  to the functional factories, interfaces and `Info` records.
