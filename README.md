# sbijax

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/sbijax.svg?colorB=black&style=flat)](https://pypi.org/project/sbijax/)

> Simulation-based inference in JAX

## About

SbiJAX implements several algorithms for simulation-based inference using
[JAX](https://github.com/google/jax), [Haiku](https://github.com/deepmind/dm-haiku) and [BlackJAX](https://github.com/blackjax-devs/blackjax).

SbiJAX so far implements

- Rejection ABC (`RejectionABC`),
- Sequential Monte Carlo ABC (`SMCABC`),
- Sequential Neural Likelihood Estimation (`SNL`)

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
