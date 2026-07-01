"""The uniform interface for trainable estimators."""

from collections.abc import Callable
from typing import NamedTuple


class Estimator(NamedTuple):
  """A trainable simulation-based inference estimator.

  A record of pure functions produced by an estimator factory (e.g.
  :func:`sbijax._src.inference.likelihood.nle.nle`). The two functions share a
  uniform contract across all trainable methods:

  - ``fit(rng_key, data, **kwargs) -> (params, info)`` trains the estimator on a
    dataset and returns the fitted parameters and a loss history.
  - ``sample(rng_key, params, observable, **kwargs) -> InferenceData`` draws
    from the approximate posterior conditioned on an observation.

  Attributes:
      fit: the training function
      sample: the posterior sampling function
  """

  fit: Callable
  sample: Callable
