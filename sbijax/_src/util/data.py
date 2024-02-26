from jax import numpy as jnp

from sbijax._src.util.dataloader import named_dataset


def stack_data(data, also_data):
    """Stack two data sets.

    Args:
        data: one data set
        also_data: another data set

    Returns:
        returns the stack of the two data sets
    """
    if data is None:
        return also_data
    if also_data is None:
        return data
    return named_dataset(*[jnp.vstack([a, b]) for a, b in zip(data, also_data)])
