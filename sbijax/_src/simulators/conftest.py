import pytest

from sbijax.simulators import (
  hyperboloid,
  mixture_model_with_distractors,
  sir,
  slcp,
  solar_dynamo,
  tree,
  two_moons,
)

_simulators = [
  hyperboloid,
  mixture_model_with_distractors,
  sir,
  slcp,
  solar_dynamo,
  tree,
  two_moons,
]

try:
  from sbijax.simulators import jansen_rit
except ImportError:
  # `jansen_rit` needs the optional `jrnmm` dependency; skip it on a base
  # install rather than failing collection for every simulator.
  pass
else:
  _simulators.append(jansen_rit)


@pytest.fixture(params=_simulators)
def simulator_model(request):
  yield request.param
