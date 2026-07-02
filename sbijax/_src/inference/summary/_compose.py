"""Compose a summary network with a downstream estimator.

A :class:`~sbijax._src.inference.summary._summary_net.SummaryNet` learns a
low-dimensional statistic of the data; it does not produce a posterior. To
infer, its summaries feed an :class:`~sbijax._src.inference._estimator.Estimator`
(NLE, NRE, ...). :func:`summarized_estimator` wires the two together so the
summary transform is applied consistently to both the training data and the
observation -- the common failure mode is forgetting to summarize the
observation at sample time, which silently conditions the estimator on data it
never saw.
"""

from jax import numpy as jnp

from sbijax._src.inference._estimator import Estimator


def summarized_estimator(estimator, summary_net, summary_params):
  """Adapt an estimator to operate on learned summaries.

  Given a *pre-fitted* summary network, returns an
  :class:`~sbijax._src.inference._estimator.Estimator` whose ``fit`` trains on
  summarized data and whose ``sample`` summarizes the observation before
  sampling. Because it returns an ``Estimator`` record it stays conformant and
  composes with :func:`~sbijax.run_sequential`.

  Fit the summary network first, then wrap the estimator::

      sn = nass(make_nass_net(2, [64, 64]))
      sn_params, _ = sn.fit(key, data)
      est = summarized_estimator(nle(prior, make_maf(2)), sn, sn_params)
      params, info = est.fit(key, data)              # trains on summaries
      idata = est.sample(key, params, y_observed)    # summarizes y_observed

  Args:
      estimator: the downstream estimator consuming the summaries
      summary_net: a fitted
          :class:`~sbijax._src.inference.summary._summary_net.SummaryNet`
      summary_params: the summary network's fitted parameters

  Returns:
      an :class:`~sbijax._src.inference._estimator.Estimator`
  """

  def fit(rng_key, data, **kwargs):
    return estimator.fit(
      rng_key, summary_net.summarize(summary_params, data), **kwargs
    )

  def sample(rng_key, params, observable, **kwargs):
    summary = summary_net.summarize(summary_params, jnp.atleast_2d(observable))
    return estimator.sample(rng_key, params, summary, **kwargs)

  return Estimator(fit=fit, sample=sample)
