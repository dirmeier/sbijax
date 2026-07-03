"""sbijax: Simulation-based inference in JAX."""

__version__ = "0.4.0"

from sbijax._src.diagnostics import sbc
from sbijax._src.diagnostics.convergence import ess, rhat
from sbijax._src.inference import Estimator, run_sequential
from sbijax._src.inference._sample_info import DirectSampleInfo, MCMCSampleInfo
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

__all__ = [
  "ABCSampler",
  "CMPEInfo",
  "DirectSampleInfo",
  "Estimator",
  "FMPEInfo",
  "MCMCSampleInfo",
  "NLEInfo",
  "NPEInfo",
  "NREInfo",
  "SummaryInfo",
  "SummaryNet",
  "abs_distance",
  "cmpe",
  "DiffEvolution",
  "ess",
  "fmpe",
  "l2_distance",
  "MultiEps",
  "nass",
  "nasss",
  "nle",
  "npe",
  "nre",
  "rhat",
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
