"""Simulator models.

Demonstrates the usage of an experimental simulator function.
"""
import functools

from jax import random as jr, numpy as jnp
from matplotlib import pyplot as plt

from sbijax.mcmc import sample_with_slice
from sbijax.simulators import mixture_model_with_distractors


def run():
  prior, simulator, likelihood = mixture_model_with_distractors()
  theta = prior.sample(seed=jr.key(0), sample_shape=(1,))
  y = simulator(jr.key(1), theta)
  y = y.at[0, :2].set(5.0)

  def joint_pdf(theta, y):
      log_prior = prior.log_prob(theta)
      log_lik = likelihood(y, theta)
      return jnp.sum(log_prior) + jnp.sum(log_lik)

  partial_joint_pdf = functools.partial(joint_pdf, y=y)
  samples = sample_with_slice(
      jr.key(2),
      partial_joint_pdf,
      prior,
  )

  _, ax = plt.subplots(figsize=(4, 4))
  ax.hist(samples['theta'].reshape(-1), color="black", bins=100)
  plt.show()


if __name__ == "__main__":
    run()
