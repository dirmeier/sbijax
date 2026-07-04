"""Sequential (multi-round) inference driver (design B)."""

# ruff: noqa: PLR0913, S610
import jax
from jax import random as jr

from sbijax._src.simulate.simulate import simulate, stack
from sbijax._src.train.sample import sample
from sbijax._src.train.train import train
from sbijax._src.util.data import flatten_chains


def _posterior_proposal(objective, params, observable, sampler):
  """Wrap the current posterior as a proposal ``(rng, n) -> theta``."""

  def proposal(rng_key, n):
    # One chain with n_samples=2n / n_warmup=n yields exactly n post-warmup
    # draws for MCMC methods and >= n for amortized; the extra kwargs are
    # ignored by amortized sample_fns.
    samples, _ = sample(
      rng_key,
      objective,
      params,
      observable,
      sampler=sampler,
      n_samples=2 * n,
      n_warmup=n,
      n_chains=1,
    )
    theta = flatten_chains(samples)
    return jax.tree_util.tree_map(lambda x: x[:n], theta)

  return proposal


def run_sequential(
  rng_key,
  objective,
  prior,
  simulator,
  observable,
  *,
  n_rounds,
  n_simulations_per_round,
  sampler=None,
  proposal_fn=None,
  **train_kwargs,
):
  """Run multi-round sequential inference.

  Round 0 simulates from the prior; later rounds simulate from a proposal built
  from the current posterior, append, and refit. NPE switches to its atomic
  objective (``objective.extra(prior)``) in rounds > 0; proposal-invariant
  methods reuse ``objective``.

  Args:
      rng_key: a jax random key
      objective: an ``ObjectiveFns``
      prior: the prior distribution
      simulator: ``(rng_key, theta) -> y``
      observable: the observation to condition on
      n_rounds: number of simulate/append/refit rounds
      n_simulations_per_round: pairs drawn each round
      sampler: a sampler (from ``make_sampler``) for MCMC-based objectives
      proposal_fn: optional ``(objective, params, observable, sampler) ->
          ((rng, n) -> theta)``; defaults to sampling the fitted posterior
      **train_kwargs: forwarded to ``train`` each round

  Returns:
      ``(params, Info)`` from the final round
  """
  if proposal_fn is None:
    proposal_fn = _posterior_proposal
  data, params, info = None, None, None
  for _ in range(n_rounds):
    sim_key, train_key, rng_key = jr.split(rng_key, 3)
    obj_r = (
      objective
      if (info is None or objective.extra is None)
      else objective.extra(prior)
    )
    proposal = (
      None
      if info is None
      else proposal_fn(objective, params, observable, sampler)
    )
    round_data = simulate(
      sim_key, prior, simulator, proposal=proposal, n=n_simulations_per_round
    )
    data = round_data if data is None else stack(data, round_data)
    params, info = train(train_key, obj_r, data, info=info, **train_kwargs)
  return params, info
