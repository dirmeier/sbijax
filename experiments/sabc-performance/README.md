# SABC performance benchmark

Compares four Simulated Annealing ABC configurations on one identical
stochastic SIR problem:

| label | library | backend |
|---|---|---|
| `sbijax` | sbijax `SABC` | JAX |
| `sabc-mlx` | sabc | MLX (Apple Silicon) |
| `sabc-numpy` | simulated_annealing_abc | NumPy/SciPy |
| `sabc-numba` | simulated_annealing_abc | Numba |

Measures speed (wall-clock mean ± standard error over 5 reps, JIT/compile
reported separately), peak host RSS, posterior-recovery bias vs the known true
`(beta, gamma)`, and distributional distance (Wasserstein-1 + energy) vs the
`sabc-numpy` reference. Every tool is allowed all the threads/cores it can use.

Each configuration runs in its own subprocess for clean per-config timing and
peak-memory readings (JAX and MLX both grab global/device resources).

## Setup

```bash
bash setup.sh            # builds .venv (Python 3.12 via uv)
```

`sbijax` is installed editable from this repo (`-e ../..`) because the SABC
under test lives on this repo's local `sabc` branch. The two comparison
libraries are installed from GitHub:

- `sabc` — https://github.com/dirmeier/sabc (triggers a C++ build via
  scikit-build-core / cmake / nanobind; Apple Silicon only)
- `simulated_annealing_abc[numba]` — https://github.com/ulzegasi/SimulatedAnnealingABC

## Run

```bash
.venv/bin/python main.py --quick   # smoke test (1 rep, 500 particles, 20k sims)
.venv/bin/python main.py           # full: 10k particles, 1M sims, 5 reps
```

Outputs land in `results/`: `table.md`, `posterior.png`, `metrics.json`, and
per-run `<label>_rep<k>.{npy,json}` (all gitignored).

## Layout

```
sir_spec.py             shared constants, prior, true theta, observed generator
adapters/
  _bench.py             timing, peak-RSS, result IO
  sbijax_adapter.py     JAX SIR sim + sbijax.SABC
  mlx_adapter.py        MLX SIR sim + sabc.run
  numpy_adapter.py      NumPy/Numba SIR sim + simulated_annealing_abc.sabc
main.py                 orchestrator: runs all configs, aggregates, reports
```
