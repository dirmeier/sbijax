"""Sequential Monte Carlo approximate Bayesian computation.

Implements the method of :cite:t:`beaumont2009adaptive` as a functional
:class:`~sbijax._src.inference.abc._sampler.ABCSampler`. The SMC core is reused
from the existing implementation and exposed behind a pure ``sample`` function.
"""

from sbijax._src.abc.smc_abc import SMCABC as _SMCABCEngine
from sbijax._src.inference.abc._sampler import ABCSampler


def smcabc(prior, simulator, summary_fn, distance_fn):
  """Construct a sequential Monte Carlo ABC sampler.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      simulator: a callable ``(rng_key, theta) -> y``
      summary_fn: maps simulated data to summary statistics
      distance_fn: distance between simulated and observed summaries

  Returns:
      an :class:`~sbijax._src.inference.abc._sampler.ABCSampler`
  """
  engine = _SMCABCEngine((prior, simulator), summary_fn, distance_fn)

  def sample(rng_key, observable, **kwargs):
    result = engine.sample_posterior(rng_key, observable, **kwargs)
    return result[0] if isinstance(result, tuple) else result

  return ABCSampler(sample=sample)
