"""NASS+NLE example.

Demonstrates neural approximate sufficient statistics with sequential
neural likelihood estimation on the simple likelihood complex posterior model.
"""

from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax import nass, nle, simulate, stack, summarized_estimator
from sbijax.nn import make_maf, make_nass_net


def prior_fn():
  prior = tfd.JointDistributionNamed(
    {"theta": tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))}, batch_ndims=0
  )
  return prior


def simulator_fn(seed, theta):
  theta = theta["theta"]
  orig_shape = theta.shape
  if theta.ndim == 2:
    theta = theta[:, None, :]
  us_key, noise_key = jr.split(seed)

  def _unpack_params(ps):
    m0 = ps[..., [0]]
    m1 = ps[..., [1]]
    s0 = ps[..., [2]] ** 2
    s1 = ps[..., [3]] ** 2
    r = jnp.tanh(ps[..., [4]])
    return m0, m1, s0, s1, r

  m0, m1, s0, s1, r = _unpack_params(theta)
  us = tfd.Normal(0.0, 1.0).sample(
    seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
  )
  xs = jnp.empty_like(us)
  xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
  y = xs.at[:, :, :, 1].set(
    s1 * (r * us[:, :, :, 0] + jnp.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
  )
  if len(orig_shape) == 2:
    y = y.reshape((*theta.shape[:1], 8))
  else:
    y = y.reshape((*theta.shape[:2], 8))
  return y


def run(n_rounds, n_iter):
  prior = prior_fn()
  y_observed = jnp.array(
    [
      [
        -0.9707123,
        -2.9461224,
        -0.4494722,
        -3.4231849,
        -0.13285634,
        -3.364017,
        -0.85367596,
        -2.4271638,
      ]
    ]
  )

  summary_net = nass(make_nass_net(5, (64, 64)))
  estimator = nle(prior, make_maf(5))

  data, params_nle, params_nass = None, None, None
  for i in range(n_rounds):
    simulate_key, nass_key, nle_key = jr.split(jr.fold_in(jr.PRNGKey(1), i), 3)

    # build proposal: summarize the observation using current summary params
    if params_nass is not None:
      s_observed = summary_net.summarize(params_nass, y_observed)
      proposal_samples, _ = estimator.sample(
        simulate_key, params_nle, s_observed
      )
      flat = proposal_samples["theta"].reshape(
        -1, proposal_samples["theta"].shape[-1]
      )

      def proposal(rng_key, n, _flat=flat):
        idx = jr.choice(rng_key, _flat.shape[0], (n,), replace=True)
        return {"theta": _flat[idx]}

      round_data = simulate(simulate_key, prior, simulator_fn, proposal=proposal, n=2_000)
    else:
      round_data = simulate(simulate_key, prior, simulator_fn, n=2_000)

    data = round_data if data is None else stack(data, round_data)

    params_nass, _ = summary_net.fit(nass_key, data, n_iter=n_iter)
    summarized_data = summary_net.summarize(params_nass, data)
    params_nle, _ = estimator.fit(nle_key, summarized_data, n_iter=n_iter)

  s_observed = summary_net.summarize(params_nass, y_observed)
  samples, _ = estimator.sample(jr.PRNGKey(3), params_nle, s_observed)
  theta = samples["theta"].reshape(-1, samples["theta"].shape[-1])
  print("posterior mean:", jnp.mean(theta, axis=0))
  print("posterior std: ", jnp.std(theta, axis=0))


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--n-iter", type=int, default=1_000)
  parser.add_argument("--n-rounds", type=int, default=15)
  args = parser.parse_args()
  run(args.n_rounds, args.n_iter)
