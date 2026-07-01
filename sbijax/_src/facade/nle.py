"""Object-oriented facade for neural likelihood estimation.

A thin wrapper over the functional :func:`~sbijax._src.inference.likelihood.nle`
core. It holds the fitted parameters so that callers can ``fit`` then ``sample``
without threading ``params`` themselves. It contains no algorithm logic.
"""

from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.mcmc import sample_with_nuts


class NLE:
  """Neural likelihood estimation.

  Args:
      prior: a ``tfd`` distribution serving as the prior over parameters
      network: a conditional density estimator exposing a ``log_prob`` method
      sampler: an MCMC sampler used to draw from the posterior; defaults to NUTS
  """

  def __init__(self, prior, network, *, sampler=sample_with_nuts):
    self._estimator = nle(prior, network, sampler=sampler)
    self._params = None

  def fit(self, rng_key, data, **kwargs):
    """Fit the estimator and retain the trained parameters.

    Args:
        rng_key: a jax random key
        data: a dataset as returned by :func:`sbijax._src.simulate.simulate`
        **kwargs: forwarded to the functional ``fit``

    Returns:
        a tuple of the fitted parameters and the loss history
    """
    self._params, info = self._estimator.fit(rng_key, data, **kwargs)
    return self._params, info

  def sample(self, rng_key, observable, **kwargs):
    """Sample from the approximate posterior using the fitted parameters.

    Args:
        rng_key: a jax random key
        observable: the observation to condition on
        **kwargs: forwarded to the functional ``sample``

    Returns:
        an inference data object of posterior samples
    """
    if self._params is None:
      raise RuntimeError("call 'fit' before 'sample'")
    return self._estimator.sample(rng_key, self._params, observable, **kwargs)
