---
title: 'Simulation-based Inference with the Python Package sbijax'
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

In natural sciences like astrophysics, biology or neuroscience, models can be frequently formulated as simulator functions, i.e., computer programs that stochastically map a parameter vector $\theta$ to synthetic data $x$. Even though simulators of this kind can represent realistic descriptions of real-world data-generating processes, they often do emit intractable probability density functions. However, statistical inference of the parameters of a simulator requires the likelihood $p(x | \theta)$ to be tractable, such that inference algorithms like Markov chain Monte Carlo can be applied.
Simulation-based inference circumvents this by only requiring the ability to sample from the simulator: parameters $\theta$ are drawn from the prior distribution $p(\theta)$, an observable $x$ is drawn from the simulator $p(x|\theta)$, and the resulting pairs are used to approximate the posterior distribution. In computational statistics, neural simulation-based inference (SBI) describes an emerging family of methods for Bayesian inference for simulator models that use neural networks as surrogate models. Here we introduce `sbijax`, a Python package that implements a wide variety of state-of-the-art methods in neural SBI using a user-friendly programming interface.
Targeted at domain scientists, e.g., in computational physics or computational biology, and SBI researchers, `sbijax` offers high-level functionality to quickly construct SBI estimators and compute posterior distributions with only a few lines of code.
In addition, the package provides functionality for conventional approximate Bayesian computation, to compute model diagnostics, and to automatically estimate summary
statistics. By virtue of being entirely written in `JAX`, `sbijax` is extremely computationally efficient, allowing rapid training of neural networks and executing code automatically in parallel on both CPU and GPU.

# Statement of Need

Modern approaches to neural simulation-based inference (SBI) utilize recent developments in neural density estimation or score-based generative modelling to build surrogate models to approximate Bayesian posterior distributions.
Similarly to conventional methods, such as approximate Bayesian computation (ABC, @sisson2018handbook) and its sequential (SMC-ABC; @sisson2007sequential, @beaumont2009adaptive) and annealing-based (e.g., SABC, @albert2015simulated) variants, neural SBI methods infer this posterior distribution by first simulating synthetic
data and then numerically constructing an appropriate approximation to this pseudo data set. SBI methods are attractive for a couple of reasons.
On the one hand, this family of methods has been shown to be more computationally efficient and often more accurate than ABC methods, in particular for smaller simulation budgets. On the other hand,
SBI allows to easily amortize inference, i.e., to infer the posterior distribution for multiple different observations once a neural model has been trained.

Here we propose `sbijax`, a Python package implementing state-of-the-art methodology of neural simulation-based inference.
While the main focus of the package is the implementation of recent algorithms to make them available to practitioners, e.g., @albert2025simulated or @gloeckler2024allinone,
`sbijax` also implements common methods from approximate Bayesian computation, e.g., SMC-ABC, to have the entire SBI toolbox in one efficient package (see Table \ref{tbl-methods} for an overview).
In addition, `sbijax` provides functionality for model diagnostics, posterior visualization and Markov Chain Monte Carlo (MCMC) sampling.

The package uses the high-performance computing framework `JAX` as a backend [@jax2018github].
Using `JAX` has several advantages, including a) that it uses the same syntax as `numpy` [@harris2020array] which enables a seamless transition for applied scientists who already are familiar with it,
and b) that empirical evaluations have shown that `JAX` can be significantly faster than `PyTorch` (see, e.g., @phan2019composable). Our package heavily builds on libraries from
the `JAX`-verse and common Bayesian inference tools. Specifically, we use `Haiku` [@haiku2020github] to construct and train neural networks, `surjectors` for normalizing flow based
density estimation [@dirmeier2024surjectors], `TensorFlow Probability` [@dillon2017tensorflow] to define statistical distributions, and `BlackJAX` [@cabezas2024blackjax] for posterior sampling using Markov Chain Monte Carlo.

| **Model**                                      | **Class name** | **Reference**           |
|------------------------------------------------|------------|-----------------------------|
| Sequential Monte Carlo ABC                     | `SMCABC`   | @beaumont2009adaptive       |
| Simulated annealing ABC                        | `SABC`     | @albert2025simulated        |
| Neural likelihood estimation                   | `NLE`      | @papamakarios2019sequential  |
| Surjective neural likelihood estimation        | `SNLE`     | @dirmeier2025simulationbased     |
| Automatic posterior transformation             | `NPE`      | @greenberg2019automatic     |
| Contrastive neural ratio estimation            | `NRE`      | @miller2022contrastive      |
| Flow matching posterior estimation             | `FMPE`     | @wildberger2023flow         |
| Posterior Score Estimation                     | `NPSE`     | @sharrock2024sequential     |
| All-In-One Posterior Estimation                | `AIO`      | @gloeckler2024allinone      |
| Consistency model posterior estimation         | `CMPE`     | @schmitt2023consistency     |
| Neural approximate sufficient statistics       | `NASS`     | @chen2021neural             |
| Neural approximate slice sufficient statistics | `NASSS`    | @chen2023learning           |

:Implemented SBI methods in `sbijax` \label{tbl-methods}.

# State of the field

While a plethora of different models has been proposed in the recent literature, the development of adequate software packages has not followed at the same pace,
and only few packages exist that allow modelers to use these methods. Most prominently, the Python package `sbi` [@tejero-cantero2020sbi]
implements several approaches for neural simulation-based inference, such as a neural posterior, likelihood-ratio, and likelihood estimation [@cranmer2020frontier] utilizing a `PyTorch` backend [@paszke2019pytorch].
The package additionally provides an API for model diagnostics, such as posterior predictive checks, effective sample size computations and simulation-based calibration.
However, the package lacks implementations of recent developments which pose the state-of-the-art in the field, such as by @chen2023learning, @dirmeier2025simulationbased or @albert2025simulated.
For approximate Bayesian computation, several Python packages are available. In particular `abcpy` [@dutta2021abcpy] implements a multitude of different ABC algorithms.
However, none of these packages implement modern (neural) SBI methods.

# Software design

`sbijax` is designed as a modular and extensible toolbox for SBI in `JAX`, combining a functional programming philosophy with a flexible interface for both expert users and domain scientists. Its design is guided by the following principles:

1) Faithful alignment with `JAX`’s functional paradigm.
`sbijax` implements a fully functional API in the idiom of `Haiku`: every method is a factory function that returns a tuple of pure functions. All non-permanent state variables, i.e., neural network parameters and optimizer states, are passed explicitly instead of being held by an object.
A neural estimator, e.g., `nle(make_maf(\cdot))`, returns training primitives and a sampling function which are consumed by the generic drivers `train` and `sample`, while an ABC method, e.g., `sabc(prior, simulator)`, returns a sampler that draws from the approximate posterior directly.
No function mutates internal state facilitating composability, and ensures compatibility with `JAX` transformations such as `jit`, `vmap`, and `pmap`.

2) Separation of inference methods and neural network implementations.
`sbijax` focuses on implementing SBI algorithms, while neural network components are defined using `Haiku`.
This allows users to construct models directly with `Haiku`, and to incorporate probabilistic building blocks from `Distrax`
 [@deepmind2020jax] and `surjectors` [@dirmeier2024surjectors].
 As a result, model definitions remain flexible and low-level, while seamlessly interoperating with the broader `JAX` ecosystem.

3) A deliberately low-level interface.
An inference workflow typically written as a short sequence of ordinary function calls, where the inputs and outputs of every step, e.g., the simulated data set and the network parameters, are managed by the developer explicitly. This asks developers to be deliberate about the workflow they are executing rather than calling the member functions of an object whose internal state is (often) opaque to them. We consider this trade-off worthwhile for a scientific software tool, since it makes a method transparent to the practitioner using it (albeit it induces some additional complexity) and straightforward to modify for a researcher extending it.

4) Support for extensibility and research.
`sbijax` is structured to facilitate experimentation with new SBI methods.
Its modular design allows components such as neural architectures, training objectives, and sampling strategies to be easily replaced or extended, making it suitable as both a research framework and a practical toolbox.
Since every method reduces to the same small set of primitives, namely an initialization, a gradient step, an evaluation step and a sampling function, a newly implemented objective can be trained and sampled from with the existing drivers without any further changes.

5) Accessibility for domain scientists.
In addition to its flexibility, `sbijax` includes pre-implemented models with sensible defaults, enabling use without in-depth expertise in deep learning.
In the simplest case, users only need to define a prior and a simulator to run inference workflows—often in as few as five lines of code.

# Research impact statement

`sbijax` has already been used extensively in the Machine Learning literature. @dirmeier2025simulationbased and @dirmeier2025causal proposed novel SBI methods for
surjective neural likelihood estimation and posterior estimation using causal constraints, respectively, where they used `sbijax` heavily for their experimental evaluations.
@ulzega2025shedding used `sbijax` to infer the posterior distribution of a complicated Bayesian model from the astrophysics literature.
@albert2025simulated developed a novel ABC method that uses `sbijax` for model evaluation.

# AI usage disclosure

No GenAI or other AI tools have been used in writing the software or this manuscript.

# Acknowledgements

This research was supported by the Swiss National Science Foundation (Grant No. $200021\_208249$).

# References
