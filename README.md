# sbijax <img src="https://raw.githubusercontent.com/dirmeier/sbijax/main/docs/_static/sticker.png" align="right" width="160px"/>

[![active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![ci](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/dirmeier/sbijax/branch/main/graph/badge.svg?token=dn1xNBSalZ)](https://codecov.io/gh/dirmeier/sbijax)
[![documentation](https://readthedocs.org/projects/sbijax/badge/?version=latest)](https://sbijax.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/sbijax.svg?colorB=black&style=flat)](https://pypi.org/project/sbijax/)

> Simulation-based inference in JAX

## About

``Sbijax`` is a Python library for neural simulation-based inference and
approximate Bayesian computation using [JAX](https://github.com/google/jax).
It implements recent methods, such as *Simulated-annealing ABC*,
*Surjective Neural Likelihood Estimation*, *Neural Approximate Sufficient Statistics*
or *Consistency model posterior estimation*, as well as methods to compute model
diagnostics and for visualizing posterior distributions.

> [!CAUTION]
> ⚠️ As per the LICENSE file, there is no warranty whatsoever for this free software tool. If you discover bugs, please report them.

## Examples

`Sbijax` implements a slim object-oriented API with functional elements stemming from
JAX. All a user needs to define is a prior model, a simulator function and an inferential algorithm.
For example, you can define a neural likelihood estimation method and generate posterior samples like this:

```python
from jax import numpy as jnp, random as jr
from sbijax import NLE
from sbijax.nn import make_maf
from tensorflow_probability.substrates.jax import distributions as tfd

def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Normal(jnp.zeros(2), jnp.ones(2))
    ), batch_ndims=0)
    return prior

def simulator_fn(seed, theta):
    p = tfd.Normal(jnp.zeros_like(theta["theta"]), 0.1)
    y = theta["theta"] + p.sample(seed=seed)
    return y


fns = prior_fn, simulator_fn
model = NLE(fns, make_maf(2))

y_observed = jnp.array([-1.0, 1.0])
data, _ = model.simulate_data(jr.PRNGKey(1))
params, _ = model.fit(jr.PRNGKey(2), data=data)
posterior, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)
```

More self-contained examples can be found in [examples](https://github.com/dirmeier/sbijax/tree/main/examples).

## Documentation

Documentation can be found [here](https://sbijax.readthedocs.io/en/latest/).

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

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
[good first issue](https://github.com/dirmeier/sbijax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

In order to contribute:

1) Clone `sbijax` and install `hatch` via `pip install hatch`,
2) create a new branch locally `git checkout -b feature/my-new-feature` or `git checkout -b issue/fixes-bug`,
3) implement your contribution and ideally a test case,
4) test it by calling `make tests`, `make lints` and `make format` on the (Unix) command line,
5) submit a PR 🙂

## Citing sbijax

If you find our work relevant to your research, please consider citing:

```
@article{dirmeier2024simulation,
  title={Simulation-based inference with the Python Package sbijax},
  author={Dirmeier, Simon and Ulzega, Simone and Mira, Antonietta and Albert, Carlo},
  journal={arXiv preprint arXiv:2409.19435},
  year={2024}
}
```

## Acknowledgements

> [!NOTE]
> 📝 The API of the package is heavily inspired by the excellent Pytorch-based [`sbi`](https://github.com/sbi-dev/sbi) package.

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
