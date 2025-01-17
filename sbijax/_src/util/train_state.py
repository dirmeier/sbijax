from typing import NamedTuple

from haiku import Params
from jax import Array
from optax import OptState


class TrainingState(NamedTuple):
    """Current configuration of a neural network training state."""

    params: Params
    opt_state: OptState
    rng_key: Array
    step: Array
