"""sbijax: Simulation-based inference in JAX."""

__version__ = "0.3.0"

import os

from matplotlib.pyplot import style

from sbijax._src.abc.smc_abc import SMCABC
from sbijax._src.cmpe import CMPE
from sbijax._src.fmpe import FMPE
from sbijax._src.nass import NASS
from sbijax._src.nasss import NASSS
from sbijax._src.nle import NLE
from sbijax._src.npe import NPE
from sbijax._src.nre import NRE
from sbijax._src.plot.plot import (
    plot_ess,
    plot_loss_profile,
    plot_posterior,
    plot_rank,
    plot_rhat_and_ress,
    plot_trace,
)
from sbijax._src.snle import SNLE
from sbijax._src.util.data import (
    as_inference_data,
    inference_data_as_dictionary,
)

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
    "plot_ess",
    "plot_rank",
    "plot_rhat_and_ress",
    "plot_loss_profile",
    "plot_posterior",
    "plot_trace",
    "as_inference_data",
    "inference_data_as_dictionary",
]

style_path = os.path.join(os.path.dirname(__file__), "_src", "plot", "styles")
style.core.USER_LIBRARY_PATHS.append(style_path)
style.core.reload_library()
style.use(os.path.join(style_path, "sbijax.mplstyle"))
del os, style
