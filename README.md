# sbijax

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sbijax/actions/workflows/ci.yaml)

> Simulation-based inference in JAX

## About

SbiJAX implements several algorithms for simulation-based inference using
[BlackJAX](https://github.com/blackjax-devs/blackjax), [Haiku](https://github.com/deepmind/dm-haiku) and [JAX](https://github.com/google/jax).

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

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
["good first issue"](https://github.com/dirmeier/sbijax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). In order to contribute:

1) Fork the repository and install `hatch` and `pre-commit`

```bash
pip install hatch pre-commit
pre-commit install
```

2) Create a new branch in your fork and implement your contribution

3) Test your contribution/implementation by calling `hatch run test` on the (Unix) command line before submitting a PR

```bash
hatch run test:lint
hatch run test:test
```

4) Submit a pull request :slightly_smiling_face:

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
