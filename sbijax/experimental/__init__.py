"""Experimental sbijax methods.

NPSE (:cite:t:`sharrock2024sequential`) and AiO (:cite:t:`gloeckler2024allinone`)
are functional factories that delegate to the ``fmpe`` core, plus a
truncated-prior proposal for sequential inference.
"""

from sbijax._src.experimental._truncated import make_truncated_proposal
from sbijax._src.experimental.aio import aio
from sbijax._src.experimental.npse import npse

__all__ = ["aio", "make_truncated_proposal", "npse"]
