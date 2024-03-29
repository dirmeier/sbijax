import abc

from jax import random as jr


# pylint: disable=too-many-instance-attributes,unused-argument,
# pylint: disable=too-few-public-methods
class SBI(abc.ABC):
    """SBI base class."""

    def __init__(self, model_fns):
        """Construct an SBI object.

        Args:
            model_fns: tuple
        """
        self.prior_sampler_fn, self.prior_log_density_fn = model_fns[0]
        self.simulator_fn = model_fns[1]
        self._len_theta = len(self.prior_sampler_fn(seed=jr.PRNGKey(123)))
