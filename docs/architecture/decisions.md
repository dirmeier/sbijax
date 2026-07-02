# Architecture Decision Log — sbijax 0.4 redesign

Decisions from the target-architecture design session. See
[architecture.md](architecture.md) for the resulting design and
[structural-assessment.md](structural-assessment.md) for the motivating
analysis.

## DR-001: Functional core with a thin OO facade

**Status**: Superseded by DR-010
**Date**: 2026-07-01
**Context**: The estimators are stateful classes that fuse many responsibilities
and diverge from the function-first idiom of optax/blackjax/distrax. We need to
modernise without discarding the familiar `.fit().sample()` ergonomics users
rely on.
**Decision**: Rewrite the internals as factories returning records of pure
functions with explicit state, and keep a thin OO facade (`NLE(...).fit()
.sample()`) that delegates to the core and contains no algorithm logic.
**Alternatives considered**:
- Fully functional, drop the OO classes — most idiomatic, but breaks every
  notebook and call site with no soft landing.
- Keep OO, only clean internals — least disruptive, but stays stateful and never
  reaches the target bar.
**Consequences**: Two layers to maintain, but the facade is trivial. The core
becomes testable in isolation and composable; users keep their muscle memory.

## DR-002: Three interfaces, not one

**Status**: Accepted
**Date**: 2026-07-01
**Context**: The method families genuinely differ — trainable estimators
(NPE/FMPE/CMPE/NLE/NRE/SNLE) train a net then sample; ABC (SABC/SMCABC) has no
neural training; NASS/NASSS learn summaries, not posteriors.
**Decision**: Commit to three interfaces — `Estimator` (fit → sample),
`ABCSampler` (sample), `SummaryNet` (fit → summarize).
**Alternatives considered**:
- One interface for everything — forces no-op slots (trivial `fit` for ABC,
  summaries through the `sample` slot) and yields a lowest-common-denominator
  seam that fits none of them well.
- Let it emerge — deferred; the families are well enough understood to commit
  now.
**Consequences**: The public surface names three concepts instead of one, but
each is honest and deep. Avoids a false uniform seam. The conformance harness
gains three contracts instead of one.

## DR-003: Breaking 0.4 release

**Status**: Accepted
**Date**: 2026-07-01
**Context**: The redesign changes construction, state handling, and package
layout. Preserving the current API would constrain the design to legacy shapes.
**Decision**: Ship a breaking 0.4 with a migration guide.
**Alternatives considered**:
- Deprecate, don't break — new API alongside old shims for a release or two;
  more work, and the author confirmed downstream breakage is acceptable.
- Must stay compatible — would freeze the very shapes we are trying to remove.
**Consequences**: Existing user code and notebooks must be migrated. Freedom to
design the right core. A migration guide is now a required deliverable.

## DR-004: MCMC sampler injected at estimator construction

**Status**: Accepted
**Date**: 2026-07-01
**Context**: Amortized methods sample the posterior directly from `params`;
likelihood/ratio methods (NLE/NRE) turn `params` into a log-density and need
MCMC to draw from it. The sampler is a real dependency for some methods and
irrelevant for others.
**Decision**: Inject the sampler when constructing a likelihood/ratio estimator
(`nle(prior, net, sampler=nuts)`), so `sample(key, params, observable)` has a
uniform signature across the whole Estimator family.
**Alternatives considered**:
- Pass the sampler at `sample` time — more per-call flexibility, but the
  `sample` signature would then differ between amortized and likelihood/ratio
  methods, breaking the uniform interface that is the point of the redesign.
**Consequences**: The MCMC kernel becomes a swappable adapter chosen once.
Changing sampler means reconstructing the estimator, which is cheap and
explicit.

## DR-005: Sequential inference as a standalone driver

**Status**: Accepted
**Date**: 2026-07-01
**Context**: Multi-round (sequential) inference currently lives inside the
estimator as a `self.n_round` counter and a bifurcated loss, coupling sequential
and amortized code paths.
**Decision**: Provide a standalone `run_sequential(key, estimator, simulator,
observable, n_rounds, ...)` driver that orchestrates simulate → append → refit,
keeping the estimator single-round and stateless.
**Alternatives considered**:
- Estimator owns rounds internally — fewer moving parts for the user, but
  re-introduces the per-estimator state and sequential/amortized coupling the
  redesign removes.
**Consequences**: Round logic concentrates in one driver (one level above
`train_loop`). The estimator loses `self.n_round`. The driver composes with the
standalone `simulate` module (DR-006).

## DR-006: Standalone simulation module; estimators drop the simulator

**Status**: Accepted
**Date**: 2026-07-01
**Context**: Data generation (`simulate_data` / `simulate_and_possibly_append`)
is currently a method on the estimator, bundling data generation with training
and inference.
**Decision**: Move data generation to a standalone `simulate(key, prior,
simulator, proposal, n)` module. Estimators consume `data`; they no longer hold
the simulator. The old `model_fns = (prior, simulator)` bundle splits — the
`simulate` module takes the pair, the estimator takes only `prior` (plus network
and sampler).
**Alternatives considered**:
- Keep simulation on the estimator — familiar, but preserves the god-object
  bundling.
**Consequences**: The estimator's dependency set shrinks to what it actually
uses. The data pipeline is independently testable. Sequential and one-shot
workflows share one simulation entry point.

## DR-007: Stateless estimators returning explicit (params, info)

**Status**: Accepted
**Date**: 2026-07-01
**Context**: Estimators currently carry mutable attributes (`self.prior`,
`self.model`, `self.n_round`, `self._prior_bijectors`), making them hard to
reason about and test.
**Decision**: `fit(key, data) -> (params, info)` returns the trained pytree and
a diagnostics record; `sample(key, params, observable) -> InferenceData` takes
`params` explicitly. No mutation in the core; the facade may hold `params` for
convenience only.
**Alternatives considered**:
- Keep params as estimator state — the current design; rejected as the source of
  the coupling and test difficulty.
**Consequences**: Fully explicit state threading, JAX-idiomatic. The facade
absorbs the small ergonomic cost of holding `params` so casual users are not
burdened.

## DR-008: Package geography by inference family

**Status**: Accepted
**Date**: 2026-07-01
**Context**: `_src/` is a flat dump of ~8 loose algorithm files next to
subpackages; the family taxonomy exists only in filenames.
**Decision**: Reorganise into `inference/{posterior,likelihood,ratio,summary}`,
plus `abc/`, `simulate/`, `train/`, `mcmc/`, `nn/`, `facade/`, `experimental/`.
**Alternatives considered**:
- Keep the flat layout — no navigation cost to change, but leaves the core
  algorithms homeless and the taxonomy implicit.
**Consequences**: Clear geography and discoverability. A large mechanical move;
scheduled late in the roadmap, after the interface stabilises, to avoid churn
during the functional rewrite.

## DR-009: Correctness harness — conformance, calibration, benchmark

**Status**: Accepted
**Date**: 2026-07-01
**Context**: Each method currently has a single smoke test checking output
shapes. Nothing proves the posteriors are correct or that methods share a
contract.
**Decision**: Build three test tiers — a registry-driven conformance suite per
interface, SBC calibration on a tractable problem, and a small sbibm-style
benchmark table.
**Alternatives considered**:
- Keep smoke tests only — cheap, but leaves correctness unverified and the
  interfaces unenforced.
**Consequences**: Higher upfront test investment; in return, every method is
provably conformant and calibrated, and refactors are de-risked. The conformance
suite should land early (roadmap step 2) so subsequent migration is guarded.

## DR-010: Fully functional API, no OO facade

**Status**: Accepted
**Date**: 2026-07-01
**Context**: DR-001 kept a thin OO facade for continuity. On reflection the
current OO API is already effectively functional at the call site (`fit` returns
`params`, `sample_posterior` takes `params`), so the facade adds a class the
author explicitly does not want and buys almost nothing. The desired idiom is
dm-haiku's `hk.transform -> Transformed(init, apply)` and blackjax's
`blackjax.nuts -> SamplingAlgorithm(init, step)` — a factory returning a
`NamedTuple` of pure functions, both libraries already in use here.
**Decision**: Drop the OO facade entirely. The public API is factory functions
(`nle`, `npe`, `nre`, ...) that return an `Estimator` `NamedTuple` of pure
functions, with parameters threaded explicitly:

```
est = nle(prior, net, sampler=nuts)     # cf. hk.transform / blackjax.nuts
params, info = est.fit(key, data)       # cf. Transformed.init
samples      = est.sample(key, params, observable)   # cf. Transformed.apply(params, ...)
```

**Alternatives considered**:
- Thin OO facade (DR-001) — rejected: the author does not want classes, and the
  surface it preserves is barely different from the functional one.
- Keep params inside the estimator (stateful `sample(key, y)`) — rejected: breaks
  the explicit-params contract that makes the design JAX-idiomatic.
**Consequences**: One layer instead of two; the public surface is a set of
factories returning `Estimator` NamedTuples, structurally identical to the
haiku/blackjax records the codebase already uses. Supersedes DR-001. The
migration guide replaces class construction with factory calls.

## DR-011: Per-method `Info` records; `round` the only read-back field

**Status**: Accepted
**Date**: 2026-07-02
**Context**: `fit` returns `(params, info)`, where `info` is currently a bare
`(n_epochs, 2)` loss array. Sequential inference (`run_sequential` + atomic NPE,
see [backlog.md](backlog.md)) needs `fit` to know whether it is training on prior
draws (round 0, maximum-likelihood loss) or proposal draws (round > 0, atomic
APT loss). Crucially, the atomic loss needs only the network, the prior, and
`num_atoms` — no proposal density has to be threaded — so the sole cross-round
signal is the round index. Separately, methods want to report richer training
diagnostics than a single loss array.
**Decision**: Replace the bare loss array with a **per-method `Info`
NamedTuple**, one per factory (`NPEInfo`, `NLEInfo`, ...), defined in the
method's own module — mirroring blackjax's per-algorithm `HMCInfo` / `NUTSInfo`.
Every `Info` exposes at least `round: int` and `losses` (the `(n_epochs, 2)`
history); methods add their own diagnostic fields (e.g. NPE's `num_atoms`).
`fit` gains an optional `info=None` argument: round 0 when `None`, otherwise
`info.round + 1`. **`round` is the only field `fit` reads on input**; every
other field is output-only diagnostics. `run_sequential` threads `info` back as
the round carry.
**Alternatives considered**:
- One shared `Info` across all methods — a coupling point: an NPE-specific
  diagnostic would land on every method's contract, reintroducing the
  god-object coupling the redesign removes (cf. the assessment's gap 2).
- A bare `round: int` kwarg on `fit` (the backlog's first sketch) — sufficient
  for atomic NPE, but leaves diagnostics a loose array with no room for
  method-specific reporting.
- blackjax-style split of `state` (carry) from `info` (diagnostics) into two
  records — cleanest in theory, but the carry here is a single int (`round`); a
  second one-field record is not worth its weight. We deliberately merge and
  document that `round` is the sole read-back field.
**Consequences**: The conformance suite pins a **structural** contract (every
`Info` has `round` + `losses` of the right shape) rather than one shared type;
method-specific fields are checked by each method's own test. `fit(key, data)`
stays unburdened for amortized single-round use (`info` defaults to `None`).
Resolves the "info contents" open question in [architecture.md](architecture.md)
and enables the sequential driver and atomic NPE (backlog item 1).
