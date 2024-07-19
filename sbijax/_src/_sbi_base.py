import abc

from jax import random as jr

from sbijax._src.util.dataloader import as_batch_iterators


# pylint: disable=too-many-instance-attributes,unused-argument,
# pylint: disable=too-few-public-methods
class SBI(abc.ABC):
    """SBI base class."""

    def __init__(self, model_fns):
        """Construct an SBI object.

        Args:
            model_fns: tuple
        """
        prior = model_fns[0]()
        self.prior_sampler_fn, self.prior_log_density_fn = (
            prior.sample,
            prior.log_prob,
        )
        self.simulator_fn = model_fns[1]
        self._len_theta = len(self.prior_sampler_fn(seed=jr.PRNGKey(123)))

    @staticmethod
    def as_iterators(
        rng_key, data, batch_size, percentage_data_as_validation_set
    ):
        """Convert the data set to an iterable for training.

        Args:
            rng_key: a jax random key
            data: a tuple with 'y' and 'theta' elements
            batch_size: the size of each batch
            percentage_data_as_validation_set: fraction

        Returns:
            two batch iterators
        """
        return as_batch_iterators(
            rng_key,
            data,
            batch_size,
            1.0 - percentage_data_as_validation_set,
            True,
        )
