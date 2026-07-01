"""Example simulators from the SBI literature."""

from sbijax._src.simulators.hyperboloid import hyperboloid
from sbijax._src.simulators.mixture_model_distractors import (
  mixture_model_with_distractors,
)
from sbijax._src.simulators.sir import sir
from sbijax._src.simulators.slcp import slcp
from sbijax._src.simulators.solar_dynamo import solar_dynamo
from sbijax._src.simulators.tree import tree
from sbijax._src.simulators.two_moons import two_moons

__all__ = [
  "hyperboloid",
  "jansen_rit",
  "mixture_model_with_distractors",
  "sir",
  "slcp",
  "solar_dynamo",
  "tree",
  "two_moons",
]


def __getattr__(name):
  """Lazily import the Jansen-Rit simulator so `jrnmm` stays optional."""
  if name == "jansen_rit":
    try:
      from sbijax._src.simulators.jansen_rit import jansen_rit  # noqa: PLC0415
    except ImportError as e:
      raise ImportError(
        "`jansen_rit` requires the optional `jrnmm` dependency; install "
        "it with `pip install sbijax[all]`."
      ) from e
    return jansen_rit
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
