"""Neural posterior score estimation (experimental).

Implements (truncated sequential) neural posterior score estimation
(:cite:t:`sharrock2024sequential`). At the estimator level NPSE is identical to
:func:`~sbijax.fmpe` -- a score network trained by the score-matching loss and
sampled by rejection -- so this factory delegates to it. The method's
distinctive truncated-prior proposal for sequential rounds is provided by
:func:`~sbijax._src.experimental._truncated.make_truncated_proposal` and passed
to :func:`~sbijax.run_sequential` via its ``proposal_fn`` hook.
"""

from sbijax._src.inference.posterior.fmpe import fmpe


def npse(network):
  """Construct a neural posterior score estimator.

  Args:
      network: a score network with ``loss``, ``sample`` and ``log_prob``
          methods (e.g. :func:`sbijax.experimental.nn.make_score_model`)

  Returns:
      an :class:`~sbijax._src.train._types.ObjectiveFns`
  """
  return fmpe(network)
