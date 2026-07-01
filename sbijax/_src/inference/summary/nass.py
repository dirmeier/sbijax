"""Neural approximate sufficient statistics.

Implements the NASS method of :cite:t:`chen2023learning` as a functional
:class:`~sbijax._src.inference.summary._summary_net.SummaryNet`. The network
learns a summary of the data by maximising a Jensen-Shannon mutual-information
bound; the JSD loss is reused from the existing implementation.
"""

from sbijax._src.inference.summary._summary_net import make_summary_net
from sbijax._src.nass import _jsd_summary_loss


def nass(network):
  """Construct a neural approximate sufficient statistics summary network.

  Args:
      network: a NASS summary network with ``forward``, ``summary`` and
          ``critic`` methods

  Returns:
      a :class:`~sbijax._src.inference.summary._summary_net.SummaryNet`
  """
  return make_summary_net(network, _jsd_summary_loss)
