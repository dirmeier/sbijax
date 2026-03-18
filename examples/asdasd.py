"""Sequential Monte Carlo ABC example.

Demonstrates sequential Monte Carlo ABC on a simple bivariate Gaussian example.
"""
import argparse

import chex
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import SMCABC, plot_posterior
from sbijax._src.simulators.slcp import slcp

prior, simulator, lik_fn = slcp()


for shape, n_obs in zip([(), (1,), (10,)], [1, 1, 10]):
  theta = prior.sample(seed=jr.key(1), sample_shape=shape)
  sim = simulator(jr.key(0), theta)
  log_lik = lik_fn(sim, theta)
  print(log_lik)
  chex.assert_rank(log_lik, 1)
  chex.assert_axis_dimension(sim, 0, n_obs)
