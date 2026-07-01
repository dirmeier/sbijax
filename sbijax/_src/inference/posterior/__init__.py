"""Neural posterior estimation methods."""

from sbijax._src.inference.posterior.cmpe import cmpe
from sbijax._src.inference.posterior.fmpe import fmpe
from sbijax._src.inference.posterior.npe import npe

__all__ = ["cmpe", "fmpe", "npe"]
