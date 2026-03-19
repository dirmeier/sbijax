import pytest

from sbijax.simulators import (
  hyperboloid,
  jansen_rit,
  mixture_model_with_distractors,
  sir,
  slcp,
  solar_dynamo,
  tree,
  two_moons,
)


@pytest.fixture(
  params=[
    hyperboloid,
    jansen_rit,
    mixture_model_with_distractors,
    sir,
    slcp,
    solar_dynamo,
    tree,
    two_moons,
  ]
)
def simulator_model(request):
  yield request.param
