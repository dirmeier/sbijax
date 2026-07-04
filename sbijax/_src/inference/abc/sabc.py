"""Simulated annealing approximate Bayesian computation.

Implements the method of :cite:t:`albert2025simulated` as a functional
``ABCSampler``. The particle-annealing
core is reused from the existing implementation; this factory exposes it behind
a pure ``sample`` function taking the prior and simulator separately.
"""

from sbijax._src.inference.abc._sabc_engine import SABC as _SABCEngine
from sbijax._src.inference.abc._sabc_engine import abs_distance
from sbijax._src.inference.abc._sampler import ABCSampler


def sabc(prior, simulator, *, summary_fn=lambda x: x, distance_fn=abs_distance):
  """Construct a simulated annealing ABC sampler.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      simulator: a callable ``(rng_key, theta) -> y``
      summary_fn: maps simulated data to summary statistics
      distance_fn: distance between simulated and observed summaries

  Returns:
      an ``ABCSampler``
  """
  engine = _SABCEngine((prior, simulator), summary_fn, distance_fn)

  def sample(rng_key, observable, **kwargs):
    return engine.sample_posterior(rng_key, observable, **kwargs)

  return ABCSampler(sample=sample)
