# sbijax <img src="https://raw.githubusercontent.com/dirmeier/sbijax/main/docs/_static/sticker.png" align="right" width="160px"/>

[![ci](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/dirmeier/sbijax/branch/main/graph/badge.svg?token=dn1xNBSalZ)](https://codecov.io/gh/dirmeier/sbijax)
[![documentation](https://readthedocs.org/projects/sbijax/badge/?version=latest)](https://sbijax.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/sbijax.svg?colorB=black&style=flat)](https://pypi.org/project/sbijax/)

> Simulation-based inference in JAX

``Sbijax`` is a Python library for neural simulation-based inference and
approximate Bayesian computation using [JAX](https://github.com/google/jax).
It implements recent methods, such as *Simulated Annealing ABC*,
*Surjective Neural Likelihood Estimation*, *Neural Approximate Sufficient Statistics*
or *Neural Posterior Score Estimation*.

> [!CAUTION]
> ⚠️ As per the LICENSE file, there is no warranty whatsoever for this free software tool. If you discover bugs, please report them.

## Quickstart

`Sbijax` implements a fully functional API in the idiom of [Haiku](https://github.com/google-deepmind/dm-haiku):
every method is a factory returning a tuple of pure functions. All a user needs to define is a prior function, a simulator function
and an inferential algorithm. For example, you can define a neural likelihood estimation method and generate posterior samples like this:

```python
from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import nle, train, sample, simulate
from sbijax.mcmc import make_sampler, nuts
from sbijax.nn import make_maf

prior = tfd.JointDistributionNamed(dict(
    theta=tfd.Normal(jnp.zeros(2), jnp.ones(2))
), batch_ndims=0)

def simulator_fn(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["theta"]), 0.1)
    y = theta["theta"] + p.sample(seed=seed)
    return y

estimator = nle(make_maf(2))

y_observed = jnp.array([-1.0, 1.0])
data = simulate(jr.key(1), prior, simulator_fn, n=10_000)
params, info = train(jr.key(2), estimator, data)
samples, _ = sample(
    jr.key(3), estimator, params, y_observed,
    sampler=make_sampler(nuts, prior=prior),
)
```

More self-contained examples can be found in [examples](https://github.com/dirmeier/sbijax/tree/main/examples).

## Workflow

Every method in `sbijax` takes the same two user-supplied inputs:

* a **prior**: needs a `.sample(seed=rng_key)` method returning a pytree, a
  batched form `.sample(seed=rng_key, sample_shape=(n,))`, and a
  `.log_prob(theta)` method accepting that same pytree structure. A pytree
  here just means a (possibly nested) dict of arrays, e.g. what the `prior`
  in the example above returns from `.sample(...)`:

  ```python
  {"theta": Array([0.62, 0.84])}
  ```

  Nothing checks that the prior is literally a
  `tensorflow_probability.substrates.jax` (`tfd`) distribution, but every
  example uses a `tfd.JointDistributionNamed`, which gives you both methods
  for free. **Each distribution in the prior must be vector-valued**, even one
  that describes a single parameter: write `tfd.Normal(jnp.zeros(1), 1.0)`,
  not `tfd.Normal(0.0, 1.0)`. One draw then has shape `(1,)` and `n` draws
  have shape `(n, 1)`
* a **simulator**: a plain function `(seed, theta) -> y`. Use those exact
  argument names (or `**kwargs`) — ABC methods (`sabc`/`smcabc`) call it by
  keyword internally. `simulate`/`run_sequential` always call it with a
  **batched** `theta` (a leading axis of size `n`), so it needs to handle a
  batch of parameter draws, not a single one. It must return `y` as a matrix
  of shape `(n, d)`, even for one-dimensional data, i.e., `(n, 1)` instead of
  `(n,)`

From these two, the same pipeline applies to every neural estimator (NLE, NPE,
FMPE, NPSE, NRE, SNLE):

```mermaid
%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 45, "rankSpacing": 50}, "themeVariables": {"fontFamily": "Helvetica, Arial, sans-serif", "fontSize": "28px"}}}%%
flowchart LR
    P([prior]) -->|simulate| D[(data)]
    M([simulator]) -->|simulate| D
    D -->|"nle/npe/fmpe/..."| O([objective])
    O -->|train| T([params])
    T -->|sample| R([posterior samples])

    classDef input fill:#e8eef7,stroke:#5b7fa6,stroke-width:1.5px,color:#1c2b3a,font-weight:600;
    classDef data fill:#fbf3e3,stroke:#c99a3c,stroke-width:1.5px,color:#3a2f1c,font-weight:600;
    classDef obj fill:#eaf3ea,stroke:#4c8c5a,stroke-width:1.5px,color:#1c3a22,font-weight:600;
    classDef out fill:#f6e8ee,stroke:#a65b82,stroke-width:1.5px,color:#3a1c2b,font-weight:600;
    class P,M input
    class D data
    class O obj
    class T,R out
```

1. `simulate(rng_key, prior, simulator, n)` draws `n` prior/simulation pairs.
2. A factory (`nle`, `npe`, `fmpe`, ...) wraps a neural network into an
   `ObjectiveFns` record of pure functions.
3. `train(rng_key, objective, data)` fits it, returning `params`.
4. `sample(rng_key, objective, params, observable)` draws posterior samples
   (MCMC-based methods also take `sampler=make_sampler(nuts, prior=prior)`).

## Implemented methods

The table lists each method with its factory function and the helper that
builds a suitable network. ABC methods take the prior and simulator directly
and need no network.

| Method                                                  | Factory                      | Network helper                                                              | Reference                 |
|---------------------------------------------------------|------------------------------|-----------------------------------------------------------------------------|---------------------------|
| Sequential Monte Carlo ABC (SMC-ABC)                    | [`smcabc`][smcabc]           | none, takes `summary_fn` and `distance_fn`                                  | Beaumont et al. (2009)    |
| Simulated annealing ABC (SABC)                          | [`sabc`][sabc]               | none, optional `summary_fn` and `distance_fn`                               | Albert et al. (2025)      |
| Neural likelihood estimation (NLE)                      | [`nle`][nle]                 | [`make_maf`][make_maf], [`make_spf`][make_spf], [`make_mdn`][make_mdn]      | Papamakarios et al. (2019) |
| Surjective neural likelihood estimation (SNLE)          | [`snle`][snle]               | [`make_maf`][make_maf] or [`make_spf`][make_spf] with `n_layer_dimensions`  | Dirmeier et al. (2023)    |
| Neural posterior estimation (NPE)                       | [`npe`][npe]                 | [`make_maf`][make_maf], [`make_spf`][make_spf], [`make_mdn`][make_mdn]      | Greenberg et al. (2019)   |
| Contrastive neural ratio estimation (NRE)               | [`nre`][nre]                 | [`make_mlp`][make_mlp], [`make_resnet`][make_resnet]                        | Miller et al. (2022)      |
| Flow matching posterior estimation (FMPE)               | [`fmpe`][fmpe]               | [`make_cnf`][make_cnf]                                                      | Wildberger et al. (2023)  |
| Neural posterior score estimation (NPSE)                | [`npse`][npse]               | [`experimental.nn.make_score_model`][make_score_model]                      | Sharrock et al. (2024)    |
| All-in-one posterior estimation (AIO)                   | [`experimental.aio`][aio]    | [`experimental.nn.make_simformer_based_score_model`][make_simformer]        | Gloeckler et al. (2024)   |
| Consistency model posterior estimation (CMPE)           | [`experimental.cmpe`][cmpe]  | [`make_cm`][make_cm]                                                        | Schmitt et al. (2023)     |
| Neural approximate sufficient statistics (NASS)         | [`nass`][nass]               | [`make_nass_net`][make_nass_net]                                            | Chen et al. (2021)        |
| Neural approximate slice sufficient statistics (NASSS)  | [`nasss`][nasss]             | [`make_nasss_net`][make_nasss_net]                                          | Chen et al. (2023)        |

NASS and NASSS learn summary statistics, not a posterior. Pass the fitted
summary network to [`summarized_estimator`][summarized_estimator] to train a
neural estimator on the summaries, or use its `summarize_fn` as the
`summary_fn` of an ABC method. The network helpers are defaults: a factory
accepts any network with the methods listed in its docstring (see
[Using Flax linen networks](https://sbijax.readthedocs.io/en/latest/notebooks/flax_linen.html)).
Methods in `sbijax.experimental` may change or be removed. Full citations are
on the [references](https://sbijax.readthedocs.io/en/latest/references.html)
page.

[smcabc]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.smcabc
[sabc]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.sabc
[nle]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.nle
[snle]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.snle
[npe]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.npe
[nre]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.nre
[fmpe]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.fmpe
[npse]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.npse
[nass]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.nass
[nasss]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.nasss
[summarized_estimator]: https://sbijax.readthedocs.io/en/latest/api/sbijax.html#sbijax.summarized_estimator
[aio]: https://sbijax.readthedocs.io/en/latest/api/sbijax.experimental.html#sbijax.experimental.aio
[cmpe]: https://sbijax.readthedocs.io/en/latest/api/sbijax.experimental.html#sbijax.experimental.cmpe
[make_score_model]: https://sbijax.readthedocs.io/en/latest/api/sbijax.experimental.html#sbijax.experimental.nn.make_score_model
[make_simformer]: https://sbijax.readthedocs.io/en/latest/api/sbijax.experimental.html#sbijax.experimental.nn.make_simformer_based_score_model
[make_maf]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_maf
[make_spf]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_spf
[make_mdn]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_mdn
[make_mlp]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_mlp
[make_resnet]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_resnet
[make_cnf]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_cnf
[make_cm]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_cm
[make_nass_net]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_nass_net
[make_nasss_net]: https://sbijax.readthedocs.io/en/latest/api/sbijax.nn.html#sbijax.nn.make_nasss_net

## Installation

Make sure to have a working `JAX` installation. Depending whether you want to use CPU/GPU/TPU,
please follow [these instructions](https://github.com/google/jax#installation).

To install from PyPI, just call the following on the command line:

```bash
pip install sbijax
```

To install the latest GitHub <RELEASE>, use:

```bash
pip install git+https://github.com/dirmeier/sbijax@<RELEASE>
```

## Documentation

Documentation can be found [here](https://sbijax.readthedocs.io/en/latest/).

## Contributing and Support

If you have questions, encounter problems, or need support with this software, please use the following channels:

* **Questions & Discussions:** For general questions, usage help, or architectural discussions, please open a new thread in our [GitHub Discussions](https://github.com/dirmeier/sbijax/discussions) tab.
* **Bug Reports & Feature Requests:** To report a bug, software problem, or suggest a new feature, please submit an issue via our [GitHub Issue Tracker](https://github.com/dirmeier/sbijax/issues). Please check existing issues before opening a new one to ensure it hasn't already been reported.

Code contributions in the form of pull requests are more than welcome. A good way to
start is to check out issues labelled
[good first issue](https://github.com/dirmeier/sbijax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). If you are unsure, if starting to work on a PR makes
sense, feel free to open an issue or discussion thread.

In order to contribute:

1) Clone `sbijax` and install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/).
2) Install all dependencies using `uv sync --all-groups`.
3) Install the Git hooks:
   ```bash
   uv run pre-commit install -t pre-commit -t commit-msg
   ```
4) Create a new branch locally, e.g. `git checkout -b feature/my-new-feature`.
5) Implement your contribution and ideally a test case.
6) Check your work (see below).
7) Submit a PR 🙂.

### Development commands

The project uses `uv` for everything:

```bash
uv sync --all-groups
uv run pre-commit run --all-files
uv run pytest # only runs fast tests
uv run pytest -m slow # runs all tests
uv run ruff check sbijax examples
uv run ruff check --fix sbijax examples
uv run ruff format sbijax examples
uv run mypy sbijax examples
```

## Acknowledgements

> [!NOTE]
> 📝 The API of the package is heavily inspired by [`Haiku`](https://github.com/google-deepmind/dm-haiku).
