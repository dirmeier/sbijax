"""sbijax: Simulation-based inference in JAX."""

__version__ = "0.3.6"

from sbijax._src.abc.sabc import (
  SABC,
  DiffEvolution,
  MultiEps,
  SingleEps,
  abs_distance,
  l2_distance,
  sq_distance,
  weighted_sq,
)
from sbijax._src.abc.smc_abc import SMCABC
from sbijax._src.cmpe import CMPE
from sbijax._src.fmpe import FMPE
from sbijax._src.nass import NASS
from sbijax._src.nasss import NASSS
from sbijax._src.nle import NLE
from sbijax._src.npe import NPE
from sbijax._src.nre import NRE
from sbijax._src.snle import SNLE
from sbijax._src.util.data import (
  as_inference_data,
  inference_data_as_dictionary,
)

__all__ = [
  "CMPE",
  "FMPE",
  "NASS",
  "NASSS",
  "NLE",
  "NPE",
  "NRE",
  "SABC",
  "SMCABC",
  "SNLE",
  "DiffEvolution",
  "MultiEps",
  "SingleEps",
  "abs_distance",
  "as_inference_data",
  "inference_data_as_dictionary",
  "l2_distance",
  "plot_ess",
  "plot_loss_profile",
  "plot_posterior",
  "plot_rank",
  "plot_rhat_and_ress",
  "plot_trace",
  "sq_distance",
  "weighted_sq",
]

_PLOT_FNS = frozenset(
  {
    "plot_ess",
    "plot_loss_profile",
    "plot_posterior",
    "plot_rank",
    "plot_rhat_and_ress",
    "plot_trace",
  }
)


def __getattr__(name):
  """Lazily import plotting helpers so matplotlib stays optional."""
  if name in _PLOT_FNS:
    try:
      from sbijax._src.plot import plot  # noqa: PLC0415
    except ImportError as e:
      raise ImportError(
        f"`{name}` requires the optional plotting dependencies; install "
        "them with `pip install sbijax[all]`."
      ) from e
    return getattr(plot, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
