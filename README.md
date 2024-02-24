# sbijax

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/sbijax.svg?colorB=black&style=flat)](https://pypi.org/project/sbijax/)

> Simulation-based inference in JAX

## About

`sbijax` implements several algorithms for simulation-based inference using
[JAX](https://github.com/google/jax), [Haiku](https://github.com/deepmind/dm-haiku),
[Distrax](https://github.com/deepmind/distrax) and [BlackJAX](https://github.com/blackjax-devs/blackjax).

`sbijax` so far implements

- [Sequential Monte Carlo ABC](https://www.routledge.com/Handbook-of-Approximate-Bayesian-Computation/Sisson-Fan-Beaumont/p/book/9780367733728) (`SMCABC`),
- [Neural Likelihood Estimation](https://arxiv.org/abs/1805.07226) (`SNL`)
- [Surjective Neural Likelihood Estimation](https://arxiv.org/abs/2308.01054) (`SSNL`)
- [Neural Posterior Estimation C](https://arxiv.org/abs/1905.07488) (short `SNP`)
- [Contrastive Neural Ratio Estimation](https://arxiv.org/abs/2210.06170) (short `SNR`)
- [Neural Approximate Sufficient Statistics](https://arxiv.org/abs/2010.10079) (`SNASS`)
- [Neural Approximate Slice Sufficient Statistics](https://openreview.net/forum?id=jjzJ768iV1) (`SNASSS`)

where the acronyms in parentheses denote the names of the methods in `sbijax`.

## Examples

You can find several self-contained examples on how to use the algorithms in `examples`.

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

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
