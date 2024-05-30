# sbijax

[![active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![ci](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/sbijax.svg?colorB=black&style=flat)](https://pypi.org/project/sbijax/)

> Simulation-based inference in JAX

## About

`sbijax` implements several algorithms for simulation-based inference in [JAX](https://github.com/google/jax), such as

- [Sequential Monte Carlo ABC](https://www.routledge.com/Handbook-of-Approximate-Bayesian-Computation/Sisson-Fan-Beaumont/p/book/9780367733728) (`SMCABC`)
- [Neural Likelihood Estimation](https://arxiv.org/abs/1805.07226) (`SNL`)
- [Surjective Neural Likelihood Estimation](https://arxiv.org/abs/2308.01054) (`SSNL`)
- [Neural Posterior Estimation C](https://arxiv.org/abs/1905.07488) (short `SNP`)
- [Contrastive Neural Ratio Estimation](https://arxiv.org/abs/2210.06170) (short `SNR`)
- [Neural Approximate Sufficient Statistics](https://arxiv.org/abs/2010.10079) (`SNASS`)
- [Neural Approximate Slice Sufficient Statistics](https://openreview.net/forum?id=jjzJ768iV1) (`SNASSS`)
- [Flow matching posterior estimation](https://arxiv.org/abs/2305.17161) (`SFMPE`)
- [Consistency model posterior estimation](https://arxiv.org/abs/2312.05440) (`SCMPE`)

where the acronyms in parentheses denote the names of the classes in `sbijax`. It builds on the Python packages [Surjectors](https://github.com/dirmeier/surjectors), [Haiku](https://github.com/deepmind/dm-haiku),
[Distrax](https://github.com/deepmind/distrax) and [BlackJAX](https://github.com/blackjax-devs/blackjax).

> [!CAUTION]
> ⚠️ As per the LICENSE file, there is no warranty whatsoever for this free software tool. If you discover bugs, please report them.

## Examples

`sbijax` uses an object-oriented API with functional elements stemming from JAX. You can, for instance, define
a neural likelihood estimation method and generate posterior samples like this:

```python
import distrax
import optax
from jax import numpy as jnp, random as jr
from sbijax import SNL
from sbijax.nn import make_affine_maf

def prior_model_fns():
    p = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    return p.sample, p.log_prob

def simulator_fn(seed, theta):
    p = distrax.Normal(jnp.zeros_like(theta), 1.0)
    y = theta + p.sample(seed=seed)
    return y

prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn
model = SNL(fns, make_affine_maf(2))

y_observed = jnp.array([-1.0, 1.0])
data, _ = model.simulate_data(jr.PRNGKey(0), n_simulations=5)
params, _ = model.fit(jr.PRNGKey(1), data=data, optimizer=optax.adam(0.001))
posterior, _ = model.sample_posterior(jr.PRNGKey(2), params, y_observed)
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
4) test it by calling `hatch run test` on the (Unix) command line,
5) submit a PR 🙂

## Citing sbijax

If you find our work relevant to your research, please consider citing:

```
@article{dirmeier2024sbijax,
    author = {Simon Dirmeier and Antonietta Mira and Carlo Albert},
    title = {Simulation-based inference with the Python Package sbijax},
    year = {2024},
}
```

## Acknowledgements

> [!NOTE]
> 📝 The API of the package is heavily inspired by the excellent Pytorch-based [`sbi`](https://github.com/sbi-dev/sbi) package.

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
