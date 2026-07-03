"""Sequential (multi-round) inference driver.

Orchestrates multi-round inference as a standalone driver rather than estimator
state (DR-005): each round simulates from the current posterior, appends to the
accumulated dataset, and refits. The round is carried across refits by the
estimator's per-method ``Info`` (DR-011); the estimator itself stays stateless.
"""

# ruff: noqa: PLR0913
import jax
from jax import random as jr

from sbijax._src.simulate.simulate import simulate, stack
from sbijax._src.util.data import flatten_chains


def _posterior_proposal(estimator, params, observable):
  """Wrap the current posterior as a proposal for the next round.

  Args:
      estimator: the estimator being fitted
      params: the parameters fitted in the current round
      observable: the observation the posterior is conditioned on

  Returns:
      a callable ``(rng_key, n) -> theta`` drawing ``n`` parameters from the
      posterior, in the pytree structure the prior and simulator use
  """

  def proposal(rng_key, n):
    # Amortized estimators return ``n_samples`` draws directly; MCMC-based ones
    # (NLE/NRE/SNLE) return ``n_chains * (n_samples - n_warmup)`` draws after a
    # warmup. One chain with ``n_samples=2n`` and ``n_warmup=n`` yields exactly
    # ``n`` post-warmup draws for the latter and at least ``n`` for the former;
    # the ``n_warmup``/``n_chains`` kwargs are ignored by amortized estimators.
    samples, _ = estimator.sample(
      rng_key, params, observable, n_samples=2 * n, n_warmup=n, n_chains=1
    )
    theta = flatten_chains(samples)
    return jax.tree_util.tree_map(lambda x: x[:n], theta)

  return proposal


def run_sequential(
  rng_key,
  estimator,
  prior,
  simulator,
  observable,
  *,
  n_rounds,
  n_simulations_per_round,
  proposal_fn=None,
  **fit_kwargs,
):
  """Run multi-round sequential inference.

  Round 0 simulates from the prior; each later round simulates from a proposal
  built from the posterior fitted in the previous round, appends to the
  accumulated dataset, and refits. The estimator selects its per-round
  behaviour from the ``Info`` threaded back into ``fit`` (e.g. NPE switches to
  its atomic loss in rounds > 0); estimators whose loss is proposal-invariant
  simply ignore it.

  Args:
      rng_key: a jax random key
      estimator: an :class:`~sbijax._src.inference._estimator.Estimator`
      prior: a ``tfd`` distribution over parameters
      simulator: a callable ``(rng_key, theta) -> y``
      observable: the observation to condition the sequential posterior on
      n_rounds: number of simulate/append/refit rounds
      n_simulations_per_round: number of pairs drawn each round
      proposal_fn: an optional ``(estimator, params, observable) -> ((rng_key,
          n) -> theta)`` factory building the next round's proposal; defaults to
          sampling the fitted posterior. Pass e.g.
          :func:`~sbijax._src.experimental._truncated.make_truncated_proposal`
          for truncated-prior proposals.
      **fit_kwargs: forwarded to ``estimator.fit`` each round

  Returns:
      a tuple of the parameters fitted in the final round and its ``Info``
  """
  if proposal_fn is None:
    proposal_fn = _posterior_proposal
  data, params, info = None, None, None
  for _ in range(n_rounds):
    sim_key, fit_key, rng_key = jr.split(rng_key, 3)
    proposal = (
      None if info is None else proposal_fn(estimator, params, observable)
    )
    round_data = simulate(
      sim_key, prior, simulator, proposal=proposal, n=n_simulations_per_round
    )
    data = round_data if data is None else stack(data, round_data)
    params, info = estimator.fit(fit_key, data, info=info, **fit_kwargs)
  return params, info
