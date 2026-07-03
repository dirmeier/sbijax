"""Surjective neural likelihood estimation.

Implements the method of :cite:t:`dirmeier2023simulation`. SNLE is identical to
NLE at the objective level; the dimensionality reduction that makes it
"surjective" is a property of the network (a surjective flow), not of the
training or sampling logic. This factory therefore delegates to :func:`nle`.
"""

from sbijax._src.inference.likelihood.nle import nle


def snle(network):
  """Construct a surjective neural likelihood objective.

  Args:
      network: a surjective conditional density estimator with a ``log_prob``
          method that reduces the dimensionality of the data

  Returns:
      an :class:`~sbijax._src.train._types.ObjectiveFns`
  """
  return nle(network)
