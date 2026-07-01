# sbijax 0.4 redesign — backlog

Remaining work after the functional core and the ten-method port. See
[architecture.md](architecture.md) and [decisions.md](decisions.md).

## 1. Sequential inference driver (`run_sequential`) + atomic NPE

**Status:** not implemented. The functional `npe` is single-round amortized
only; the atomic multi-round objective (`_proposal_posterior_log_prob`,
`num_atoms`) and the round loop that lived in `_ne_base` are not ported. This is
a capability gap versus the old `NPE`.

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
data, params = None, None
for r in range(n_rounds):
    proposal = None if r == 0 else _posterior_proposal(estimator, params, observable)
    round_data = simulate(key, prior, simulator, proposal=proposal,
                          n=n_simulations_per_round)
    data = round_data if data is None else stack(data, round_data)
    params, info = estimator.fit(key, data, round=r, **fit_kwargs)
return params, info
```

Two seams make this work:

- **`estimator.fit(..., round=0)`** gains an optional `round` argument
  (default 0). NLE/NRE ignore it — their loss is proposal-invariant. NPE reads
  it: `round == 0` uses the maximum-likelihood loss (with event-space
  bijections), `round > 0` uses the atomic proposal-posterior loss. The atomic
  loss needs only the prior (already captured) and `num_atoms` (a factory
  argument), so no proposal density has to be threaded into the loss.
- **`_posterior_proposal(estimator, params, observable)`** returns a callable
  `(rng_key, n) -> theta` that draws from the current posterior. For amortized
  posterior methods this wraps `estimator.sample`; for NLE/NRE it wraps their
  MCMC `sample`. It lives next to the driver, not on the estimator.

### Open questions

- Whether `round` on `fit` is the cleanest signal, or a dedicated
  `estimator.fit_round` variant. `round` keeps one `fit` signature and is
  preferred unless it forces awkward branching.
- Proposal handoff for MCMC-sampled methods (NLE/NRE): drawing `n` proposal
  parameters per round via MCMC may be expensive; consider caching or reusing
  chains.
- SNPE truncation (as in the experimental NPSE/AiO) vs atomic correction —
  decide whether truncated proposals are a separate driver option.

## 2. Public API cutover

Wire the new factories into `sbijax/__init__.py` (`nle`, `npe`, `nre`, `fmpe`,
`cmpe`, `snle`, `sabc`, `smcabc`, `nass`, `nasss`, plus `simulate`, `stack`, and
the interface records). This is the breaking 0.4 surface (DR-003, DR-010).

## 3. Experimental methods

`experimental/npse.py` and `experimental/aio.py` currently subclass the OO
`FMPE`. When the old classes are removed they must be ported to functional
factories (they add truncated-prior sampling — see candidate 4 in the original
assessment) or moved behind the new `fmpe` core.

## 4. Correctness harness — remaining tiers

The conformance tier exists (every method checked against its interface). Still
to add (DR-009):

- **SBC calibration** — simulation-based calibration rank tests on a tractable
  problem, proving the posteriors are calibrated.
- **Benchmark table** — a small sbibm-style table proving recovery of known
  posteriors for a couple of reference tasks.

## 5. Docs and examples

- Migration guide (old class API → new functional API) for the 0.4 release.
- Update `examples/` and the estimator docstrings to the functional API.
- Sphinx reference for the `inference/`, `abc/`, `summary/`, `simulate/` trees.
