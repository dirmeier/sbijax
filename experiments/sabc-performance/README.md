# SABC performance benchmark

Compares four SABC implementations — `sbijax` (JAX), `sabc-mlx` (MLX),
`sabc-numpy`, `sabc-numba` — on three SBI tasks (`gaussian_mixture`,
`mixture_distractors`, `two_moons`), scored against an MCMC reference posterior
(TFP slice sampler on each task's tractable likelihood).

## Usage

```bash
bash setup.sh
source .venv/bin/activate

python main.py references
python main.py run
```

`run` executes each algorithm once per task (seed 0), scoring wall time
(JIT/compile excluded), peak RSS, and Wasserstein/energy distance to the
reference.

## Results

### gaussian_mixture

| algorithm | wall s (mean±se) | compile s | peak RSS MB | bias L2 | W1 vs ref | energy vs ref |
|---|---|---|---|---|---|---|
| sbijax | 0.35±0.00 | 1.88 | 551 | 0.4210 | 0.1891 | 0.1237 |
| sabc-mlx | 1.98±0.00 | 0.00 | 52 | 0.3991 | 0.1901 | 0.1234 |
| sabc-numpy | 0.50±0.00 | 0.00 | 138 | 0.4208 | 0.1956 | 0.1275 |
| sabc-numba | 0.38±0.00 | 0.46 | 185 | 0.4121 | 0.1907 | 0.1234 |

### mixture_distractors

| algorithm | wall s (mean±se) | compile s | peak RSS MB | bias L2 | W1 vs ref | energy vs ref |
|---|---|---|---|---|---|---|
| sbijax | 0.58±0.00 | 2.01 | 568 | 1.1650 | 0.3003 | 0.2265 |
| sabc-mlx | 3.60±0.00 | 0.00 | 51 | 1.0791 | 0.3261 | 0.2223 |
| sabc-numpy | 1.07±0.00 | 0.00 | 146 | 1.1544 | 0.3093 | 0.2162 |
| sabc-numba | 0.95±0.00 | 0.49 | 194 | 1.1258 | 0.2934 | 0.2066 |

### two_moons

| algorithm | wall s (mean±se) | compile s | peak RSS MB | bias L2 | W1 vs ref | energy vs ref |
|---|---|---|---|---|---|---|
| sbijax | 0.32±0.00 | 1.54 | 546 | 0.0085 | 0.0189 | 0.0354 |
| sabc-mlx | 1.40±0.00 | 0.00 | 52 | 0.0049 | 0.0206 | 0.0387 |
| sabc-numpy | 0.50±0.00 | 0.00 | 137 | 0.0012 | 0.0242 | 0.0456 |
| sabc-numba | 0.39±0.00 | 0.45 | 186 | 0.0030 | 0.0212 | 0.0398 |
