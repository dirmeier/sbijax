# 👋 Welcome to `sbijax`!

*Simulation-based inference in JAX*

`Sbijax` is a Python library for neural simulation-based inference and approximate Bayesian computation using [JAX](https://github.com/google/jax).
It implements recent methods, such as 

- [Sequential Monte Carlo ABC](https://www.routledge.com/Handbook-of-Approximate-Bayesian-Computation/Sisson-Fan-Beaumont/p/book/9780367733728) (`SMCABC`)
- [Neural Likelihood Estimation](https://arxiv.org/abs/1805.07226) (`SNL`)
- [Surjective Neural Likelihood Estimation](https://arxiv.org/abs/2308.01054) (`SSNL`)
- [Neural Posterior Estimation C](https://arxiv.org/abs/1905.07488) (short `SNP`)
- [Contrastive Neural Ratio Estimation](https://arxiv.org/abs/2210.06170) (short `SNR`)
- [Neural Approximate Sufficient Statistics](https://arxiv.org/abs/2010.10079) (`SNASS`)
- [Neural Approximate Slice Sufficient Statistics](https://openreview.net/forum?id=jjzJ768iV1) (`SNASSS`)
- [Flow matching posterior estimation](https://arxiv.org/abs/2305.17161) (`SFMPE`)
- [Consistency model posterior estimation](https://arxiv.org/abs/2312.05440) (`SCMPE`)

as well as methods to compute model diagnostics and visualization of posterior distributions.

!!! warning 

    ⚠️ As per the LICENSE file, there is no warranty whatsoever for this free software tool. If you discover bugs, please report them.

## Example

`Sbijax` uses a slim object-oriented API with functional elements stemming from JAX. All a user needs to define is a prior model,
a simulator function and an inferential algorithm. For example, you can define a neural likelihood estimation method and generate posterior samples like this:

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

## Installation

You can install `sbijax` from PyPI using:

```bash
pip install sbijax
```

To install the latest GitHub <RELEASE>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/sbijax@<RELEASE>
```

See also the installation instructions for [JAX](https://github.com/google/jax), if you plan to use `sbijax` on GPU/TPU.

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
["good first issue"](https://github.com/dirmeier/sbijax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

In order to contribute:

1) Clone `sbijax` and install `hatch` via `pip install hatch`,
2) create a new branch locally `git checkout -b feature/my-new-feature` or `git checkout -b issue/fixes-bug`,
3) implement your contribution and ideally a test case,
4) test it by calling `make tests` and `make lints` on the (Unix) command line,
5) submit a PR 🙂

## Citing `sbijax`

If you find our work relevant to your research, please consider citing:

```
@article{dirmeier2024sbijax,
    author = {Simon Dirmeier and Antonietta Mira and Carlo Albert},
    title = {Simulation-based inference with the Python Package sbijax},
    year = {2024},
}
```

## Acknowledgements

!!! note

    📝 The API of the package is heavily inspired by the excellent Pytorch-based [`sbi`](https://github.com/sbi-dev/sbi) package.

## License

`sbijax` is licensed under the Apache 2.0 License.
