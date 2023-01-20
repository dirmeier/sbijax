# sbi

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/sbi/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sbi/actions/workflows/ci.yaml)

> Simulation-based inference in JAX

## About


## Example usage

TODO

## Installation

Make sure to have a working `JAX` installation. Depending whether you want to use CPU/GPU/TPU,
please follow [these instructions](https://github.com/google/jax#installation).

To install the latest GitHub <RELEASE>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/sbi@<RELEASE>
```


## Contributing

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
["good first issue"](https://github.com/ramsey-devs/ramsey/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). In order to contribute:

1) Fork the repository and install `hatch` and `pre-commit`

```bash
pip install hatch pre-commit
pre-commit install
```

2) Create a new branch in your fork and implement your contribution

3) Test your contribution/implementation by calling `hatch run test` on the (Unix) command line before submitting a PR

```bash
hatch run test
```

4) Submit a pull request :slightly_smiling_face:

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
