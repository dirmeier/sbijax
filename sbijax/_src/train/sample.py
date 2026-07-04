"""Free sampling driver, symmetric with ``fit`` (design B)."""


def sample(rng_key, objective, params, observable, *, sampler=None, **kwargs):
  """Draw posterior samples from a trained objective.

  A one-line dispatch to ``objective.sample_fn`` kept for symmetry with
  :func:`sbijax.train`.

  Args:
      rng_key: a jax random key
      objective: an ``ObjectiveFns``
      params: the trained parameters
      observable: the observation to condition on
      sampler: a sampler from :func:`~sbijax.mcmc.make_sampler`
          (required for MCMC methods, ignored by amortized methods)
      **kwargs: forwarded to ``sample_fn``

  Returns:
      ``(samples, info)``
  """
  return objective.sample_fn(
    rng_key, params, observable, sampler=sampler, **kwargs
  )
