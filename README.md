# sbijax

[![active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![ci](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/sbijax.svg?colorB=black&style=flat)](https://pypi.org/project/sbijax/)

> Simulation-based inference in JAX

## About

`sbijax` implements several algorithms for simulation-based inference in
[JAX](https://github.com/google/jax) using [Haiku](https://github.com/deepmind/dm-haiku),
[Distrax](https://github.com/deepmind/distrax) and [BlackJAX](https://github.com/blackjax-devs/blackjax). Specifically, `sbijax` implements

- [Sequential Monte Carlo ABC](https://www.routledge.com/Handbook-of-Approximate-Bayesian-Computation/Sisson-Fan-Beaumont/p/book/9780367733728) (`SMCABC`),
- [Neural Likelihood Estimation](https://arxiv.org/abs/1805.07226) (`SNL`)
- [Surjective Neural Likelihood Estimation](https://arxiv.org/abs/2308.01054) (`SSNL`)
- [Neural Posterior Estimation C](https://arxiv.org/abs/1905.07488) (short `SNP`)
- [Contrastive Neural Ratio Estimation](https://arxiv.org/abs/2210.06170) (short `SNR`)
- [Neural Approximate Sufficient Statistics](https://arxiv.org/abs/2010.10079) (`SNASS`)
- [Neural Approximate Slice Sufficient Statistics](https://openreview.net/forum?id=jjzJ768iV1) (`SNASSS`)

where the acronyms in parentheses denote the names of the methods in `sbijax`.

## Examples

You can find several self-contained examples on how to use the algorithms in [examples](https://github.com/dirmeier/sbijax/tree/main/examples).

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

## Acknowledgements

> 📝 The package draws significant inspiration from the excellent Pytorch-based [`sbi`](https://github.com/sbi-dev/sbi) package which is substantially more
feature-complete and user-friendly, and better documented.

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
