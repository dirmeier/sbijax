"""sbijax: Simulation-based inference in JAX."""

__version__ = "0.4.0"

from sbijax._src.diagnostics.convergence import ess, rhat
from sbijax._src.diagnostics.sbc import sbc
from sbijax._src.inference._estimator import Estimator
from sbijax._src.inference.abc._sabc_engine import (
  DiffEvolution,
  MultiEps,
  SingleEps,
  abs_distance,
  l2_distance,
  sq_distance,
  weighted_sq,
)
from sbijax._src.inference.abc._sampler import ABCSampler
from sbijax._src.inference.abc.sabc import sabc
from sbijax._src.inference.abc.smcabc import smcabc
from sbijax._src.inference.likelihood.nle import nle
from sbijax._src.inference.likelihood.snle import snle
from sbijax._src.inference.posterior.cmpe import cmpe
from sbijax._src.inference.posterior.fmpe import fmpe
from sbijax._src.inference.posterior.npe import npe
from sbijax._src.inference.ratio.nre import nre
from sbijax._src.inference.sequential import run_sequential
from sbijax._src.inference.summary._compose import summarized_estimator
from sbijax._src.inference.summary._summary_net import SummaryInfo, SummaryNet
from sbijax._src.inference.summary.nass import nass
from sbijax._src.inference.summary.nasss import nasss
from sbijax._src.simulate.simulate import simulate, stack

__all__ = [
  "ABCSampler",
  "Estimator",
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
