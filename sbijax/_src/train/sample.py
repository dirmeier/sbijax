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
      prior: the prior of the model. The amortized estimators reject draws
          outside its support and return the flattened parameter vector
          under a single ``"theta"`` key; passing the prior here also
          reshapes them into the prior's pytree, matching what the MCMC and
          ABC methods return. Without it no draw is rejected and the flat
          layout is preserved.
      **kwargs: forwarded to ``sample_fn``

  Returns:
      ``(samples, info)``
  """
  samples, info = objective.sample_fn(
    rng_key, params, observable, sampler=sampler, prior=prior, **kwargs
  )
  if prior is not None:
    samples = unravel_draws(samples, prior)
  return samples, info
