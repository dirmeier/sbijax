"""Surjective neural likelihood estimation.

Implements the method of :cite:t:`dirmeier2023simulation`. SNLE is identical to
NLE at the estimator level; the dimensionality reduction that makes it
"surjective" is a property of the network (a surjective flow), not of the
training or sampling logic. This factory therefore delegates to :func:`nle`.
"""

from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.mcmc import sample_with_nuts


def snle(prior, network, *, sampler=sample_with_nuts):
  """Construct a surjective neural likelihood estimator.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      network: a surjective conditional density estimator with a ``log_prob``
          method that reduces the dimensionality of the data
      sampler: an MCMC sampler used to draw from the posterior; defaults to NUTS

  Returns:
      an :class:`~sbijax._src.inference._estimator.Estimator`
  """
  return nle(prior, network, sampler=sampler)
