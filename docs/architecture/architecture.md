# sbijax 0.4 — Target Architecture

## Overview

sbijax is a JAX library for simulation-based inference (SBI). This document
defines the target architecture for a 0.4 redesign that moves the library from
stateful estimator classes to a **fully functional API** of composable pure
functions behind a **small set of uniform interfaces**, and a dedicated
**conformance + calibration** test harness.

The public surface follows dm-haiku (`hk.transform -> Transformed(init, apply)`)
and blackjax (`blackjax.nuts -> SamplingAlgorithm(init, step)`): a factory
returns a `NamedTuple` of pure functions, with parameters threaded explicitly.
There are no classes.

The design deliberately mirrors an idiom already proven in the sibling `tqe`
library — every method is a factory returning a record of pure functions — and
brings sbijax in line with the function-first conventions of optax, blackjax, and
distrax.

See [structural-assessment.md](structural-assessment.md) for the analysis that
motivated this work.

## Context

**Problem.** The current estimators are stateful god-objects: each class fuses
the loss, parameter init, training loop, data-simulation pipeline, posterior
sampling, and MCMC wiring, and shares behaviour through a four-level inheritance
chain with thin, leaky base classes. This makes methods hard to compose, hard to
test in isolation, and stylistically out of step with the JAX ecosystem.

**Constraints and decisions** (see [decisions.md](decisions.md)):

- Fully functional API — factories returning `NamedTuple` records of pure
  functions, no OO facade (DR-010, superseding DR-001).
- Three honest interfaces rather than one forced abstraction (DR-002).
- A breaking 0.4 release is acceptable; a migration guide will accompany it
  (DR-003).
- The MCMC sampler is injected when a likelihood/ratio estimator is constructed
  (DR-004).
- Sequential inference is a standalone driver, not estimator state (DR-005).
- Data simulation is a standalone module; estimators consume data and no longer
  hold the simulator (DR-006).
- Estimators are stateless: `fit` returns explicit `(params, info)` (DR-007).

**Assumptions.**

- The prior is a constructed `tfd.Distribution` passed directly (already true on
  `main` after the simulator-seam change), not a zero-arg factory.
- Networks continue to come from the existing `nn/` factory functions.
- `InferenceData` (arviz) remains the posterior sample container.

## Components

The functional core is a set of small modules, each with a single
responsibility. Estimator factories compose them; they are not related by
inheritance.

```mermaid
flowchart TB
  subgraph core["functional core (pure)"]
    prior["prior\n(tfd.Distribution)"]
    simulate["simulate\n(data pipeline)"]
    nn["nn factories\n(networks)"]
    objective["objectives\n(pure loss fns)"]
    train["train_loop"]
    mcmc["mcmc samplers\n(run_blackjax)"]
    estimator["estimator factories\n(fit, sample)"]
    sequential["run_sequential\n(round driver)"]
  end

  subgraph interfaces["interfaces"]
    E["Estimator\nfit -> sample"]
    A["ABCSampler\nsample"]
    S["SummaryNet\nfit -> summarize"]
  end

  public["public API\nnle / npe / ... factories\n-> Estimator NamedTuples"]
  harness["conformance + SBC + benchmarks"]

  prior --> estimator
  nn --> objective
  objective --> estimator
  train --> estimator
  mcmc --> estimator
  estimator -->|implements| E
  prior --> simulate
  simulate --> sequential
  sequential --> estimator
  prior --> A
  simulate --> A
  nn --> S

  E --> public
  A --> public
  S --> public
  E -.tested by.-> harness
  A -.tested by.-> harness
  S -.tested by.-> harness
```

### Core modules

| Module | Responsibility | Owns |
|--------|----------------|------|
| `simulate` | Draw `(theta, y)` from prior or a proposal and run the simulator; stack datasets. | Nothing (pure). |
| `objectives` | Per-method pure loss functions `(params, rng, **batch) -> scalar`. | Nothing. |
| `train_loop` | Generic epoch loop: optimizer step, weighted train/val loss, early stopping, best-param tracking. Already extracted. | Nothing. |
| `mcmc` (`run_blackjax`) | Turn a log-density into posterior draws via a BlackJAX kernel. Already extracted. | Nothing. |
| `estimator factories` | Compose prior + network + objective (+ sampler) into an `Estimator` record. | Closure over its dependencies; no mutable state. |
| `run_sequential` | Orchestrate multi-round inference: simulate from the current proposal, append, refit. | The round loop; no estimator state. |

### Interfaces

Three interfaces, because the families genuinely differ (DR-002):

```mermaid
classDiagram
  class Estimator {
    +fit(key, data, *, info, optimizer, n_iter) (params, Info)
    +sample(key, params, observable, *, n_samples) InferenceData
  }
  class ABCSampler {
    +sample(key, observable, *, summary, distance) InferenceData
  }
  class SummaryNet {
    +fit(key, data, *, optimizer, n_iter) (params, info)
    +summarize(params, data) summaries
  }
  Estimator <|.. NPE
  Estimator <|.. FMPE
  Estimator <|.. CMPE
  Estimator <|.. NLE
  Estimator <|.. SNLE
  Estimator <|.. NRE
  ABCSampler <|.. SABC
  ABCSampler <|.. SMCABC
  SummaryNet <|.. NASS
  SummaryNet <|.. NASSS
```

- **Estimator** — trainable posterior/likelihood/ratio methods. `sample` has a
  uniform signature across the family; for NLE/NRE it runs the injected sampler
  internally, for amortized methods that argument is simply unused.
- **ABCSampler** — no neural training. Constructed with `(prior, simulator)`,
  exposes `sample` only; it simulates during sampling.
- **SummaryNet** — learns summary statistics rather than a posterior. `summarize`
  preprocesses data that is then fed to an Estimator.

### Public API

The public surface is the factory functions themselves — no wrapper layer. A
factory returns an `Estimator` (or `ABCSampler`/`SummaryNet`) `NamedTuple` of
pure functions, exactly as `hk.transform` returns `Transformed(init, apply)` and
`blackjax.nuts` returns `SamplingAlgorithm(init, step)`. Parameters are threaded
explicitly by the caller.

```
est = nle(prior, net, sampler=nuts)          # cf. hk.transform / blackjax.nuts
params, info = est.fit(key, data)            # cf. Transformed.init
samples      = est.sample(key, params, y_obs)  # cf. Transformed.apply(params, ...)
```

## Data Architecture

The central data entities and where they live:

```mermaid
erDiagram
  PRIOR ||--o{ DATASET : "draws theta"
  SIMULATOR ||--o{ DATASET : "produces y"
  DATASET ||--|| PARAMS : "fit produces"
  PARAMS ||--o{ INFERENCEDATA : "sample produces"
  NETWORK ||--|| PARAMS : "parameterises"

  DATASET {
    array y
    pytree theta
  }
  PARAMS {
    pytree weights
  }
  INFERENCEDATA {
    array posterior
    dict diagnostics
  }
```

**Ownership and flow.** No component owns mutable state. `simulate` produces a
`Dataset`; `Estimator.fit` consumes it and produces `params` plus a per-method
`Info` record; `Estimator.sample` consumes `params` and an observation and
produces `InferenceData`. `params` is a plain pytree threaded explicitly by the
caller — this is the core of the stateless design (DR-007).

`Info` is a per-method `NamedTuple` (`NPEInfo`, `NLEInfo`, ...) defined in the
method's own module, mirroring blackjax's per-algorithm `HMCInfo` / `NUTSInfo`
(DR-011). Every `Info` exposes at least `round: int` and `losses` (the
`(n_epochs, 2)` history); methods add their own diagnostic fields. `fit` takes
an optional `info=None`: `None` is round 0, otherwise the round is
`info.round + 1`. **`round` is the only field `fit` reads on input**; the rest
is output-only diagnostics. Threading `info` back is what lets `run_sequential`
carry the round across refits (see below and backlog item 1).

### Amortized fit → sample

```mermaid
sequenceDiagram
  participant U as caller
  participant Sim as simulate
  participant Est as estimator
  participant TL as train_loop
  U->>Sim: simulate(key, prior, simulator, n)
  Sim-->>U: data
  U->>Est: fit(key, data, optimizer, n_iter)
  Est->>TL: train_loop(loss_fn, data, ...)
  TL-->>Est: best_params, losses
  Est-->>U: params, info
  U->>Est: sample(key, params, y_obs)
  Est-->>U: InferenceData
```

### Sequential (multi-round)

```mermaid
sequenceDiagram
  participant U as caller
  participant Seq as run_sequential
  participant Sim as simulate
  participant Est as estimator
  U->>Seq: run_sequential(key, estimator, simulator, y_obs, n_rounds)
  loop each round
    Seq->>Sim: simulate(key, prior, simulator, proposal)
    Sim-->>Seq: round_data
    Seq->>Est: fit(key, all_data, info=info)
    Est-->>Seq: params, info
    Note over Seq: info.round advances; proposal := posterior(params, y_obs)
  end
  Seq-->>U: params, info
```

## API & Integration

Contracts are described, not implemented.

**Data pipeline**

- `simulate(rng_key, prior, simulator, *, proposal=None, n) -> Dataset` — draws
  `theta` from `proposal or prior`, runs `simulator`, returns `{y, theta}`.
- `stack(data, new_data) -> Dataset` — append datasets across rounds.

**Estimator** (constructed per method)

- `npe(prior, net, *, num_atoms=10) -> Estimator`
- `nle(prior, net, *, sampler=nuts) -> Estimator`
- `nre(prior, net, *, sampler=nuts, num_classes=...) -> Estimator`
- `Estimator.fit(rng_key, data, *, info=None, optimizer=None, n_iter,
  batch_size, n_early_stopping_patience, n_early_stopping_delta) -> (params,
  Info)` — `info` defaults to `None` (round 0); pass the previous round's `Info`
  to advance the round. Returns a per-method `Info` NamedTuple (DR-011).
- `Estimator.sample(rng_key, params, observable, *, n_samples, **sampler_kwargs)
  -> InferenceData`

**ABCSampler**

- `sabc(prior, simulator, *, ...) -> ABCSampler`
- `ABCSampler.sample(rng_key, observable, *, summary, distance, ...) ->
  InferenceData`

**SummaryNet**

- `nass(prior, net) -> SummaryNet`
- `SummaryNet.fit(rng_key, data, *, ...) -> (params, info)`
- `SummaryNet.summarize(params, data) -> summaries`

**Sequential driver**

- `run_sequential(rng_key, estimator, simulator, observable, *, n_rounds,
  n_sims_per_round, ...) -> (params, info)`

**Communication pattern.** All calls are synchronous, pure JAX. The only
"injected dependencies" are the sampler (into likelihood/ratio estimators) and
the network/prior (into every factory) — each a swappable adapter chosen at
construction (DR-004).

## Cross-Cutting Concerns

**Correctness harness** (the credibility layer, DR-009). Three tiers:

1. *Conformance* — a registry-driven test that every Estimator/ABCSampler/
   SummaryNet satisfies its interface contract on a toy problem: return shapes,
   `InferenceData` structure, and the **structural `Info` contract** (every
   `Info` exposes `round: int` and a `(n_epochs, 2)` `losses` array; DR-011).
   Method-specific `Info` fields are checked by that method's own test, not the
   shared suite. Mirrors `tqe`'s `test_objective_contract.py`.
2. *Calibration* — simulation-based calibration (SBC) rank tests proving the
   posteriors are calibrated, run on a small analytically-tractable problem.
3. *Benchmark* — a small sbibm-style table proving recovery of known posteriors
   for a couple of reference tasks.

**Package geography** (DR-008). Group by inference family instead of the current
flat `_src/` dump:

```
_src/
  inference/
    posterior/   # npe, fmpe, cmpe
    likelihood/  # nle, snle
    ratio/       # nre
    summary/     # nass, nasss
  abc/           # sabc, smcabc
  simulate/      # data pipeline
  train/         # train_loop
  mcmc/          # run_blackjax + kernels
  nn/            # network factories
  experimental/  # npse, aio, simformer
```

The `inference/`, `abc/`, and `summary/` factories are the public API; there is
no separate facade package.

**Determinism / RNG.** Every entry point takes an explicit `rng_key`; no global
state. This is enforced by the stateless design and checked by the conformance
suite.

## Open Questions

- **`info` contents.** *Resolved (DR-011).* `Info` is a per-method NamedTuple
  exposing at least `round: int` and a `(n_epochs, 2)` `losses` array; methods
  add their own diagnostic fields. Sampling diagnostics stay in `sample`'s
  `InferenceData` (`sample_stats`), and SBC ranks stay in the calibration
  harness — neither rides in `Info`.
- **Sequential proposal construction.** *Resolved.* `run_sequential` builds the
  proposal from the fitted posterior via a `proposal_fn` hook
  (`_posterior_proposal` by default). Truncated-prior proposals (NPSE/AiO) are a
  drop-in `proposal_fn` (`experimental.make_truncated_proposal`).
- **SNLE / NASS composition.** SNLE (surjective NLE) and the SummaryNet →
  Estimator pipeline need a documented composition pattern once the core lands.
```
