"""sbijax: Simulation-based inference in JAX."""

__version__ = "0.4.0"

from sbijax._src.diagnostics import sbc
from sbijax._src.inference import Estimator, run_sequential
from sbijax._src.inference.abc import ABCSampler, sabc, smcabc
from sbijax._src.inference.abc._sabc_engine import (
  DiffEvolution,
  MultiEps,
  SingleEps,
  abs_distance,
  l2_distance,
  sq_distance,
  weighted_sq,
)
from sbijax._src.inference.likelihood import NLEInfo, nle, snle
from sbijax._src.inference.posterior import (
  CMPEInfo,
  FMPEInfo,
  NPEInfo,
  cmpe,
  fmpe,
  npe,
)
from sbijax._src.inference.ratio import NREInfo, nre
from sbijax._src.inference.summary import (
  SummaryInfo,
  SummaryNet,
  nass,
  nasss,
  summarized_estimator,
)
from sbijax._src.simulate import simulate, stack
from sbijax._src.util.data import (
  as_inference_data,
  inference_data_as_dictionary,
)

__all__ = [
  "ABCSampler",
  "CMPEInfo",
  "Estimator",
  "FMPEInfo",
  "NLEInfo",
  "NPEInfo",
  "NREInfo",
  "SummaryInfo",
  "SummaryNet",
  "abs_distance",
  "as_inference_data",
  "cmpe",
  "DiffEvolution",
  "fmpe",
  "inference_data_as_dictionary",
  "l2_distance",
  "MultiEps",
  "nass",
  "nasss",
  "nle",
  "npe",
  "nre",
  "plot_ess",
  "plot_loss_profile",
  "plot_posterior",
  "plot_rank",
  "plot_rhat_and_ress",
  "plot_trace",
  "run_sequential",
  "sabc",
  "sbc",
  "SingleEps",
  "simulate",
  "smcabc",
  "snle",
  "sq_distance",
  "stack",
  "summarized_estimator",
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
