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

## Quick start

`Sbijax` implements a fully functional API in the idiom of [Haiku](https://github.com/google-deepmind/dm-haiku):
every method is a factory returning a record of pure functions, with parameters
threaded explicitly. All a user needs to define is a prior, a simulator function
and an inferential algorithm. For example, you can define a neural likelihood
estimation method and generate posterior samples like this:

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
> 📝 The API of the package is heavily inspired by [`Haiku`](https://github.com/google-deepmind/dm-haiku).
