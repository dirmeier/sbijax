# `sbijax`

The top-level module, `sbijax`, contains all implemented methods for neural 
simulation-based inference and approximate Bayesian inference as well as
functionality for visualization.

## Posterior estimation

::: sbijax.CMPE
    options:
      members:
      - fit
      - simulate_data
      - simulate_data_and_possibly_append
      - sample_posterior

::: sbijax.FMPE
    options:
      members:
      - fit
      - simulate_data
      - simulate_data_and_possibly_append
      - sample_posterior

::: sbijax.NPE
    options:
      members:
      - fit
      - simulate_data
      - simulate_data_and_possibly_append
      - sample_posterior

## Likelihood estimation

::: sbijax.NLE
    options:
      members:
      - fit
      - simulate_data
      - simulate_data_and_possibly_append
      - sample_posterior

::: sbijax.SNLE
    options:
      members:
      - fit
      - simulate_data
      - simulate_data_and_possibly_append
      - sample_posterior

## Likelihood-ratio estimation

::: sbijax.NRE
    options:
      members:
      - fit
      - simulate_data
      - simulate_data_and_possibly_append
      - sample_posterior

## Approximate Bayesian computation

::: sbijax.SMCABC
    options:
      members: 
      - sample_posterior

## Summary statistics

::: sbijax.NASS
    options:
      members:
      - fit
      - summarize

::: sbijax.NASSS
    options:
      members:
      - fit
      - summarize

## Visualization

::: sbijax.plot_posterior
::: sbijax.plot_trace