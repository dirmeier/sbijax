"""Compose a summary network with a downstream estimator.

A :class:`~sbijax._src.train._types.SummaryFns` learns a low-dimensional
statistic of the data; it does not produce a posterior. To infer, its summaries
feed a downstream :class:`~sbijax._src.train._types.ObjectiveFns` (NLE, NRE,
...). :func:`summarized_estimator` wires the two together so the summary
transform is applied consistently to both the training batches and the
observation -- the common failure mode is forgetting to summarize the
observation at sample time, which silently conditions the estimator on data it
never saw.
"""

from jax import numpy as jnp

from sbijax._src.train._types import ObjectiveFns, TrainFns


def summarized_estimator(estimator, summary_net, summary_params):
  """Adapt an objective to operate on learned summaries.

  Given a *pre-fitted* summary network, returns an
  :class:`~sbijax._src.train._types.ObjectiveFns` whose training summarizes each
  batch before delegating to the wrapped estimator and whose ``sample_fn``
  summarizes the observation before sampling. Because it returns an
  ``ObjectiveFns`` record it stays conformant and is driven by the generic
  :func:`~sbijax._src.train.fit.fit` / sampling helpers.

  Fit the summary network first, then wrap the estimator::

      sn = nass(make_nass_net(2, [64, 64]))
      sn_params, _ = train(key, sn, data)
      est = summarized_estimator(nle(make_maf(2)), sn, sn_params)
      params, info = train(key, est, data)           # trains on summaries
      samples, _ = est.sample_fn(key, params, y_observed, sampler=sampler)

  Args:
      estimator: the downstream :class:`~sbijax._src.train._types.ObjectiveFns`
          consuming the summaries
      summary_net: a fitted
          :class:`~sbijax._src.train._types.SummaryFns`
      summary_params: the summary network's fitted parameters

  Returns:
      an :class:`~sbijax._src.train._types.ObjectiveFns`
  """

  def _summarize_batch(batch):
    ret = dict(batch)
    ret["y"] = summary_net.summarize_fn(summary_params, batch["y"])
    return ret

  inner = estimator.train

  def init_fn(optimizer, rng_key, batch):
    return inner.init_fn(optimizer, rng_key, _summarize_batch(batch))

  def step_fn(optimizer, rng_key, state, batch):
    return inner.step_fn(optimizer, rng_key, state, _summarize_batch(batch))

  def eval_fn(rng_key, state, batch):
    return inner.eval_fn(rng_key, state, _summarize_batch(batch))

  def sample_fn(rng_key, params, observable, *, sampler=None, **kwargs):
    summary = summary_net.summarize_fn(
      summary_params, jnp.atleast_2d(observable)
    )
    return estimator.sample_fn(
      rng_key, params, summary, sampler=sampler, **kwargs
    )

  return ObjectiveFns(TrainFns(init_fn, step_fn, eval_fn), sample_fn)
