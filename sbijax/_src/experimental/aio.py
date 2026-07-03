"""All-in-one simulation-based inference (experimental).

Implements all-in-one posterior estimation (:cite:t:`gloeckler2024allinone`).
As in NPSE the estimator is a score network trained by the score-matching loss
and sampled by rejection, so this factory delegates to :func:`~sbijax.fmpe`; the
difference is the network -- a transformer (simformer) score model with a mask
encoding the conditional dependencies (e.g.
:func:`sbijax.experimental.nn.make_simformer_based_score_model`). Like NPSE it
composes with :func:`~sbijax.run_sequential` and the truncated-prior proposal.

This implementation infers the joint posterior of all latent variables only (no
marginals or arbitrary conditionals), so a single fixed mask is used throughout.
"""

from sbijax._src.inference.posterior.fmpe import fmpe


def aio(network):
  """Construct an all-in-one posterior estimator.

  Args:
      network: a simformer-based score network with ``loss``, ``sample`` and
          ``log_prob`` methods

  Returns:
      an :class:`~sbijax._src.train._types.ObjectiveFns`
  """
  return fmpe(network)
