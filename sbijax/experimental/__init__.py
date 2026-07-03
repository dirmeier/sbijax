"""Experimental sbijax methods.

CMPE (:cite:t:`schmitt2023con`) and AiO (:cite:t:`gloeckler2024allinone`) are
functional objective factories, plus a truncated-prior proposal for sequential
inference.
"""

from sbijax._src.experimental._truncated import make_truncated_proposal
from sbijax._src.experimental.aio import aio
from sbijax._src.experimental.cmpe import cmpe

__all__ = ["aio", "cmpe", "make_truncated_proposal"]
