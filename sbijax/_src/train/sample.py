"""Free sampling driver, symmetric with ``fit`` (design B)."""

from sbijax._src.util.data import unravel_draws


# ruff: noqa: PLR0913
def sample(
  rng_key, objective, params, observable, *, sampler=None, prior=None, **kwargs
):
  """Draw posterior samples from a trained objective.

  A thin dispatch to ``objective.sample_fn`` kept for symmetry with
  :func:`sbijax.train`.

  Args:
      rng_key: a jax random key
      objective: an ``ObjectiveFns``
      params: the trained parameters
      observable: the observation to condition on
      sampler: a sampler from :func:`~sbijax.mcmc.make_sampler`
          (required for MCMC methods, ignored by amortized methods)
      prior: the prior the draws should be named after. The amortized
          estimators return the flattened parameter vector under a single
          ``"theta"`` key, since they never see the prior; passing it here
          reshapes them into the prior's pytree, matching what the MCMC and
          ABC methods return. Without it the flat layout is preserved.
      **kwargs: forwarded to ``sample_fn``

  Returns:
      ``(samples, info)``
  """
  # consumed here rather than forwarded: every amortized sample_fn ends in
  # **kwargs and would silently swallow it
  samples, info = objective.sample_fn(
    rng_key, params, observable, sampler=sampler, **kwargs
  )
  if prior is not None:
    samples = unravel_draws(samples, prior)
  return samples, info
