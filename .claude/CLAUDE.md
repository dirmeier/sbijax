# CLAUDE.md

Operational guide for an agent editing **sbijax** — a JAX library of
neural simulation-based inference and ABC methods. The README covers what it
is; this file covers how to change it well.

## Architecture

Every inference method is a class constructed from
`model_fns = (prior_fn, simulator_fn)` (`self.prior = prior_fn()`,
`self.simulator_fn = model_fns[1]`):

- **ABC methods** (`SABC`, `SMCABC`) subclass `SBI`
  (`sbijax/_src/_sbi_base.py`) and implement
  `sample_posterior(rng_key, observable, ...)`.
- **Neural methods** (`NPE`, `SNLE`, `CMPE`, `NASS`, …) subclass `NE`
  (`sbijax/_src/_ne_base.py`), which adds a `network`, the `simulate_*` data
  helpers, `fit(...)`, and `sample_posterior(...)`.

A new method subclasses one of these and matches those signatures — mirror an
existing method (e.g. `npe.py`) rather than inventing a new shape.

## Commands

- `make format` — ruff `--fix` + format. Run before reading a diff or committing.
- `make tests` — `uv run pytest`.
- `make lints` — `uv run ruff check`.
- **Full gate:** `uv run pre-commit run --all-files`. Run this before declaring
  work done — it is the **only** place mypy runs; `make tests`/`make lints`
  skip type-checking.

## Python style

Write code to the standard that would pass review at a top lab: **PEP 8 + the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**
(Anthropic publishes no separate guide; its public repos follow these + ruff +
full typing). ruff and mypy are the authority — run `make format`; consult
`pyproject.toml [tool.ruff]` for the full rule set. The concrete rules:

- **Naming.** `lower_with_under` for modules/functions/variables, `CapWords`
  for classes, `CAPS_WITH_UNDER` for constants, `_leading_underscore` for
  internal. Descriptive names — no abbreviations; avoid single chars except
  `i/j/k` loop indices, `e` exceptions, `f` files.
- **Docstrings.** Google-style on every public symbol: a one-line summary
  ending in a period, then `Args:` / `Returns:` (or `Yields:`) / `Raises:`
  for functions and `Attributes:` for classes.
- **Type annotations.** Annotate new and edited code — parameters and return
  types on public APIs. Use `X | None` (not `Optional[X]`); don't annotate
  `self`/`cls`; use `from __future__ import annotations` for forward refs.
  (Most existing modules are still untyped and are being migrated; don't add
  them to your diff, but everything you write should be typed.)
- **Imports.** One per line, grouped future → stdlib → third-party → local and
  sorted (ruff's isort enforces this); no relative imports.
- **Errors.** No `assert` in library code — raise an exception instead
  (`ValueError` for bad preconditions). Never bare `except:` or catch
  `Exception` except to re-raise; manage resources with `with`.
- **Functions.** Small and focused (~40 lines is a smell to refactor past);
  no mutable default arguments (use `None`); guard scripts with
  `if __name__ == "__main__":`.
- **Idioms.** Comprehensions / generator expressions over `map`/`filter` +
  lambda for simple cases; f-strings for formatting; comments explain *why*,
  not *what*.
- **Project overrides:** 2-space indentation (ruff `indent-width = 2`, not
  Google's 4) and an 80-char line limit — ruff-format owns both; don't
  hand-format.

## What to avoid (JAX traps)

- Don't use `jax.random.categorical` for N-from-N resampling — it builds an
  `(N, N)` Gumbel matrix (O(N²) time and memory). Use inverse-CDF
  (`cumsum` + `searchsorted`).
- jit hot loops and reuse the compiled function; a bare `lax.scan` re-invoked
  each call still re-traces.
- When timing, force materialization (`jax.block_until_ready` or convert to
  numpy) — async dispatch makes naive timers read ~0.
- Wrap priors in `tfd.Independent` so `log_prob` reduces to `(N,)`, not
  `(N, 1)` — the latter silently breaks downstream reductions.
