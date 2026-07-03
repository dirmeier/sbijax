"""The uniform interface for trainable estimators."""

from collections.abc import Callable
from typing import NamedTuple


def next_round(info):
  """Return the round index a ``fit`` call should train under.

  ``fit`` reads only the ``round`` field of an incoming ``Info`` (DR-011): the
  first round passes ``info=None`` and trains round 0; every later round passes
  the previous round's ``Info`` and advances by one.

  Args:
      info: the previous round's per-method ``Info``, or ``None`` for round 0

  Returns:
      ``0`` if ``info`` is ``None``, otherwise ``info.round + 1``
  """
  return 0 if info is None else info.round + 1


class Estimator(NamedTuple):
  """A trainable simulation-based inference estimator.

  A record of pure functions produced by an estimator factory (e.g.
  :func:`sbijax._src.inference.likelihood.nle.nle`). The two functions share a
  uniform contract across all trainable methods:

  - ``fit(rng_key, data, **kwargs) -> (params, info)`` trains the estimator on a
    dataset and returns the fitted parameters and a loss history.
  - ``sample(rng_key, params, observable, **kwargs) -> (samples, info)`` draws
    from the approximate posterior conditioned on an observation, returning the
    named posterior pytree and a per-method sampling record.

  Attributes:
      fit: the training function
      sample: the posterior sampling function
  """

  fit: Callable
  sample: Callable
