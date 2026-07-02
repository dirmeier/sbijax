"""Neural posterior estimation methods."""

from sbijax._src.inference.posterior.cmpe import CMPEInfo, cmpe
from sbijax._src.inference.posterior.fmpe import FMPEInfo, fmpe
from sbijax._src.inference.posterior.npe import NPEInfo, npe

__all__ = ["CMPEInfo", "FMPEInfo", "NPEInfo", "cmpe", "fmpe", "npe"]
