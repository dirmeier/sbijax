"""Neural approximate slice sufficient statistics.

Implements the NASSS method of :cite:t:`chen2021neural` as a functional
:class:`~sbijax._src.inference.summary._summary_net.SummaryNet`. It differs from
NASS only in the loss (a slice-based JSD bound with a secondary summary); the
training and summarization logic are shared.
"""

from sbijax._src.inference.summary._summary_net import make_summary_net
from sbijax._src.nasss import _jsd_summary_loss


def nasss(network):
  """Construct a neural approximate slice sufficient statistics summary network.

  Args:
      network: a NASSS summary network with ``forward``, ``summary``,
          ``secondary_summary`` and ``critic`` methods

  Returns:
      a :class:`~sbijax._src.inference.summary._summary_net.SummaryNet`
  """
  return make_summary_net(network, _jsd_summary_loss)
