"""Neural approximate sufficient statistics with neural likelihood estimation.

A self-contained example of chaining a learned summary network into a
downstream estimator with the functional 0.4 API: the 8-d data is reduced to a
2-d summary by ``nass``, and ``nle`` infers the posterior from those summaries.
:func:`sbijax.summarized_estimator` keeps the summary transform consistent
between training and the observation.
"""

from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import nass, nle, sample, simulate, summarized_estimator, train
from sbijax.mcmc import make_sampler, nuts
from sbijax.nn import make_maf, make_nass_net


def prior_fn():
  return tfd.JointDistributionNamed(
    {"theta": tfd.Normal(jnp.zeros(2), 1.0)}, batch_ndims=0
  )


def simulator_fn(seed, theta):
  # 8-d data carrying a 2-d signal plus noise
  noise = tfd.Normal(0.0, 1.0).sample((theta["theta"].shape[0], 8), seed=seed)
  return jnp.tile(theta["theta"], (1, 4)) + noise


def run():
  prior = prior_fn()
  data = simulate(jr.key(0), prior, simulator_fn, n=5_000)

  summary_net = nass(make_nass_net(2, [64, 64]))
  summary_params, _ = train(jr.key(1), summary_net, data)

  estimator = summarized_estimator(
    nle(make_maf(2)), summary_net, summary_params
  )
  params, info = train(jr.key(2), estimator, data)
  print(f"trained for {info.losses.shape[0]} epochs")

  y_observed = jnp.tile(jnp.array([-1.0, 1.0]), 4)
  samples, _ = sample(
    jr.key(3),
    estimator,
    params,
    y_observed,
    sampler=make_sampler(nuts, prior=prior),
  )
  theta = samples["theta"].reshape(-1, samples["theta"].shape[-1])
  print("posterior mean:", jnp.mean(theta, axis=0))


if __name__ == "__main__":
  run()
