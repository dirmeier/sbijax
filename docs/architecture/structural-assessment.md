# sbijax structural assessment

A zoom-out on package structure, written before the target-architecture design.
Scope is the shape of the library, not individual refactors.

## One-sentence diagnosis

sbijax is a **collection of stateful algorithm classes**. A JAX/DeepMind-grade
version is a **small set of composable, functional primitives behind one uniform
interface, backed by a conformance-and-calibration test harness.** The distance
between those two sentences is the project.

The reference architecture already exists in the sibling `tqe` library: every
objective is a factory returning
`ObjectiveFns(TrainFns(step_fn, eval_fn), sample_fn, extra)`, with one factory
seam and a `test_objective_contract.py` that every method must satisfy. sbijax
should port an idiom the author already trusts, not invent a new one.

## The big structural gaps

### 1. The dominant idiom fights the JAX ecosystem

optax, blackjax, and distrax are function-first: `init` / `update` / `step`
returning explicit state, no hidden mutation. sbijax estimators are stateful
classes carrying `self.prior`, `self.model`, `self.n_round`,
`self._prior_bijectors`. Mutable-object style is the single largest divergence
from the target bar and the root cause of why the methods are hard to compose
and test.

### 2. Each estimator is a god object

One class owns the loss, the parameter init, the (now-extracted) training loop,
the data-simulation/append pipeline, the posterior sampling, and the MCMC
wiring. These concerns change at different rates and belong in separate modules
composed together, not fused into one class and shared by inheritance.

### 3. Inheritance is deep and the bases are leaky

Four-level chains exist (`CMPE → FMPE → NE → SBI`), with `NPSE`/`AiO` hanging off
`FMPE` and `SNLE` off `NLE`. Meanwhile `SBI` is essentially a tuple-unpacker and
`NE` carries pass-through abstract methods. Deep inheritance with thin bases is
the hardest thing to navigate and the easiest to break. This wants composition.

### 4. The package geography is a flat dump

`_src/` holds ~8 loose algorithm files (`nle`, `nre`, `npe`, `fmpe`, `cmpe`,
`nass`, `nasss`, `snle`) next to the `abc/`, `mcmc/`, `nn/`, `simulators/`,
`util/` subpackages. The core algorithms — the reason the library exists — have
no home of their own, and the family taxonomy (posterior / likelihood / ratio /
summary / ABC) lives only implicitly in filenames.

### 5. Correctness infrastructure is thin

Almost every method has a single smoke test that checks output shapes. A
top-tier SBI library has three layers instead:

- a shared **conformance** suite every method passes (present in `tqe`, absent
  here),
- statistical **calibration** via SBC (simulation-based calibration) proving the
  posteriors are actually calibrated,
- a small **benchmark** table (sbibm-style) proving the methods recover known
  posteriors.

This is what turns "it runs" into "it is trusted."

### 6. Naming and discoverability

The public surface is abbreviation soup (NLE/NRE/NPE/FMPE/CMPE/NASS/NASSS/SNLE/
SABC/SMCABC) with no unifying namespace or taxonomy, `experimental/`
intermixed, and internal `_ne_base` / `_sbi_base` names. Last to fix, not first,
but real.

## What is already good (leave alone)

- Optional-dependency handling via module `__getattr__` (plotting, `jrnmm`).
- The `nn/` factory modules.
- Sphinx docs with runnable examples.

The weak points are structure and correctness-infra, not packaging or docs.

## Roadmap (ordered by leverage, lowest risk first)

1. **Define the one uniform interface** every method satisfies — the keystone.
   Adapt the `ObjectiveFns`/factory idiom to SBI's needs (amortized vs
   sequential vs ABC). Everything hangs off this; do it carefully and write it
   down.
2. **Add a conformance test** every method must pass against that interface.
   Cheap, and it de-risks every later refactor.
3. **Tracer-bullet one method** (NLE, simplest) into the new functional shape end
   to end; prove the interface; then migrate the rest one at a time behind the
   conformance suite.
4. **Reorganize the geography** by inference family once the interface is stable
   (mechanical, do it late).
5. **Add SBC calibration + a small benchmark table** — the credibility layer.
6. **API / naming / docs polish** last, once the structure underneath is
   settled.

## Next step

Run the interview-driven architecture design on step 1 and produce a
target-architecture document with diagrams and a decision log, then turn it into
issues.
