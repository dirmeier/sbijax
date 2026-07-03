"""The uniform interface for likelihood-free ABC samplers."""

from collections.abc import Callable
from typing import NamedTuple


class ABCSampler(NamedTuple):
  """A likelihood-free approximate Bayesian computation sampler.

  ABC methods do not train a network, so unlike an
  :class:`~sbijax._src.inference._estimator.Estimator` they expose only a
  ``sample`` function that simulates during sampling.

  Attributes:
      sample: ``(rng_key, observable, **kwargs) -> (particles, info)``
  """

  sample: Callable
