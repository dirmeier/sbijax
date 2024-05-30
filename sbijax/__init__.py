"""sbijax: Simulation-based inference in JAX."""

__version__ = "1.0.0"

from sbijax._src.abc.smc_abc import SMCABC
from sbijax._src.cmpe import CMPE
from sbijax._src.fmpe import FMPE
from sbijax._src.nass import NASS
from sbijax._src.nasss import NASSS
from sbijax._src.nle import NLE, SNLE
from sbijax._src.npe import NPE
from sbijax._src.nre import NRE
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.data import flatten as inference_data_as_dictionary
from sbijax._src.util.plot import plot_posterior, plot_trace

__all__ = [
    "SMCABC",
    "CMPE",
    "FMPE",
    "NASS",
    "NASSS",
    "NLE",
    "SNLE",
    "NPE",
    "NRE",
    "plot_posterior",
    "plot_trace",
    "as_inference_data",
    "inference_data_as_dictionary",
]
