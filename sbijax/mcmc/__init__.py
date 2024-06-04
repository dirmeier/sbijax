"""MCMC module."""

from sbijax._src.mcmc.irmh import sample_with_imh
from sbijax._src.mcmc.mala import sample_with_mala
from sbijax._src.mcmc.nuts import sample_with_nuts
from sbijax._src.mcmc.rmh import sample_with_rmh
from sbijax._src.mcmc.slice import sample_with_slice
from sbijax._src.mcmc.util import as_inference_data

__all__ = [
    "as_inference_data",
    "sample_with_imh",
    "sample_with_mala",
    "sample_with_rmh",
    "sample_with_nuts",
    "sample_with_slice",
]
