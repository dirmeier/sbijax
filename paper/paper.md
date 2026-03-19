---
title: 'Simulation-based inference with the Python Package sbijax'
tags:
  - Python
  - Simulation-based Inference
  - Approximate Bayesian Computation
  - Machine learning
authors:
  - name: Simon Dirmeier
    affiliation: "1, 2"
  - name: Antonietta Mira
    affiliation: "3, 4"
  - name: Carlo Albert
    affiliation: "5"
affiliations:
  - name: Swiss Data Science Center, Zurich, Switzerland
    index: 1
  - name: ETH Zurich, Zurich, Switzerland
    index: 2
  - name: Università della Svizzera italiana, Switzerland
    index: 3
  - name: University of Insubria, Italy
    index: 4
  - name: Swiss Federal Institute of Aquatic Science and Technology, Switzerland
    index: 5
date: 19 March 2026
bibliography: paper.bib
---

# Summary

Neural simulation-based inference (SBI) describes an emerging family of methods for Bayesian inference for simulator models that use neural networks as surrogate models.
Here we introduce `sbijax`, a Python package that implements a wide variety of state-of-the-art methods in neural simulation-based inference using a user-friendly
programming interface. sbijax offers high-level functionality to quickly construct SBI estimators, and compute and visualize posterior distributions with only a few lines of code.
In addition, the package provides functionality for conventional approximate Bayesian computation, to compute model diagnostics, and to automatically estimate summary
statistics. By virtue of being entirely written in `JAX`, sbijax is extremely computationally efficient, allowing rapid training of neural networks and executing code automatically in parallel on both CPU and GPU.

# Statement of Need

Modern approaches to neural simulation-based inference (SBI) utilize recent developments in neural density estimation or score-based generative modelling to build surrogate models to approximate Bayesian posterior distributions.
Similarly to conventional methods, such as approximate Bayesian computation (ABC) and its sequential (SMC-ABC) and annealing-based (e.g., SABC) variants, neural SBI methods infer this posterior distribution by first simulating synthetic
data and then numerically constructing an appropriate approximation to this pseudo data set. SBI methods are attractive for a couple of reasons.
On the one hand, this family of methods has been shown to be more computationally efficient and often more accurate than ABC methods, in particular for smaller simulation budgets. On the other hand,
SBI allows to easily amortize inference, i.e., to infer the posterior distribution for multiple different observations once a neural model has been trained.

Here we propose `sbijax`, a Python package implementing state-of-the-art methodology of neural simulation-based inference.
While the main focus of the package is the implementation of recent algorithms to make them available to practitioners, e.g., @wildberger2023flow or @schmitt2023consistency,
`sbijax` also implements common methods from approximate Bayesian computation, e.g., SMC-ABC [@beaumont2009adaptive], to have the entire SBI toolbox in one efficient package (see Table 1 for an overview).
In addition, `sbijax` provides functionality for model diagnostics, posterior visualization and Markov Chain Monte Carlo (MCMC) sampling.
The package uses the high-performance computing framework `JAX` as a backend [@jax2018github].
Using `JAX` has several advantages, including a) that it uses the same syntax as `numpy` [@harris2020array] which enables a seamless transition for applied scientists who already are familiar with it,
and b) that empirical evaluations have shown that `JAX` can be significantly faster than `PyTorch` (see, e.g., @phan2019composable).

| **Model**                                      | **Class name** | **Reference**           |
|------------------------------------------------|------------|-----------------------------|
| Sequential Monte Carlo ABC                     | `SMCABC`   | @beaumont2009adaptive       |
| Neural likelihood estimation                   | `NLE`      | @papamakarios2019sequential  |
| Surjective neural likelihood estimation        | `SNLE`     | @dirmeier2023simulation     |
| Automatic posterior transformation             | `NPE`      | @greenberg2019automatic     |
| Contrastive neural ratio estimation            | `NRE`      | @miller2022contrastive      |
| Flow matching posterior estimation             | `FMPE`     | @wildberger2023flow         |
| Posterior Score Estimation                     | `NPSE`     | @sharrock2024sequential     |
| All-In-One Posterior Estimation                | `AIO`      | @gloeckler2024allinone      |
| Consistency model posterior estimation         | `CMPE`     | @schmitt2023consistency     |
| Neural approximate sufficient statistics       | `NASS`     | @chen2021neural             |
| Neural approximate slice sufficient statistics | `NASSS`    | @chen2023learning           |

:Implemented SBI methods in `sbijax`.

# State of the field

While a plethora of different models has been proposed in the recent literature, the development of adequate software packages has not followed at the same pace,
and only few packages exist that allow modelers to use these methods. Most prominently, the Python package `sbi` [@tejero-cantero2020sbi]
implements several approaches for neural simulation-based inference, such as a neural posterior, likelihood-ratio, and likelihood estimation [@cranmer2020frontier] utilizing a `PyTorch` backend [@paszke2019pytorch].
The package additionally provides an API for model diagnostics, such as posterior predictive checks, effective sample size computations and simulation-based calibration.
However, the package lacks implementations of recent developments which pose the state-of-the-art in the field, such as by @chen2023learning, @dirmeier2023simulation or @schmitt2023consistency.
Also, by virtue of being developed in `PyTorch` it is potentially restrictive to practitioners that do not have experience with it.
For approximate Bayesian computation, several `Python` packages are available. In particular `abcpy` [@dutta2021abcpy] implements a multitude of different ABC algorithms.
However, none of these packages implement modern (neural) SBI methods.

# Research impact statement

`sbijax` has already been used extensively in the Machine Learning literature. @dirmeier2023simulation have proposed a novel method for simulation-based inference using dimensionality reduction. They use `sbijax` for their experimental section.
are using `sbijax` for their experiments. @dirmeier2025causal proposed a novel for *causal* posterior estimation where they used `sbijax` for their experimental evaluations.
@albert2025simulated developed a novel ABC method that uses `sbijax` for model evaluation.

# AI usage disclosure

No GenAI or other AI tools have been used in writing the software or this manuscript.

# Acknowledgements

This research was supported by the Swiss National Science Foundation (Grant No. $200021\_208249$).

# References
