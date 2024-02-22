from collections import namedtuple

import chex
from jax import lax
from jax import numpy as jnp
from jax import random as jr

named_dataset = namedtuple("named_dataset", "y theta")


# pylint: disable=missing-class-docstring,too-few-public-methods
class DataLoader:
    # noqa: D101
    def __init__(
        self, num_batches, idxs=None, get_batch=None, batches=None
    ):  # noqa: D107
        self.num_batches = num_batches
        self.idxs = idxs
        if idxs is not None:
            self.num_samples = len(idxs)
        else:
            self.num_samples = self.num_batches * batches[0]["y"].shape[0]
        self.get_batch = get_batch
        self.batches = batches

    def __call__(self, idx, idxs=None):  # noqa: D102
        if self.batches is not None:
            return self.batches[idx]

        if idxs is None:
            idxs = self.idxs
        return self.get_batch(idx, idxs)


# pylint: disable=missing-function-docstring
def as_batch_iterators(
    rng_key: chex.PRNGKey, data: named_dataset, batch_size, split, shuffle
):
    """Create two data batch iterators from a data set.

    Args:
        rng_key: random key
        data: a named tuple containing all dat
        batch_size: batch size
        split: fraction of data to use for training data set. Rest is used
            for validation data set.
        shuffle: shuffle the data set or no

    Returns:
        two iterators
    """
    n = data.y.shape[0]
    n_train = int(n * split)

    if shuffle:
        idxs = jr.permutation(rng_key, jnp.arange(n))
        data = named_dataset(*[el[idxs] for _, el in enumerate(data)])

    y_train = named_dataset(*[el[:n_train] for el in data])
    y_val = named_dataset(*[el[n_train:] for el in data])
    train_rng_key, val_rng_key = jr.split(rng_key)

    train_itr = as_batch_iterator(train_rng_key, y_train, batch_size, shuffle)
    val_itr = as_batch_iterator(val_rng_key, y_val, batch_size, shuffle)

    return train_itr, val_itr


# pylint: disable=missing-function-docstring
def as_batch_iterator(
    rng_key: chex.PRNGKey, data: named_dataset, batch_size, shuffle
):
    """Create a data batch iterator from a data set.

    Args:
        rng_key: random key
        data: a named tuple containing all dat
        batch_size: batch size
        shuffle: shuffle the data set or no

    Returns:
        an iterator
    """
    n = data.y.shape[0]
    if n < batch_size:
        num_batches = 1
        batch_size = n
    elif n % batch_size == 0:
        num_batches = int(n // batch_size)
    else:
        num_batches = int(n // batch_size) + 1

    idxs = jnp.arange(n)
    if shuffle:
        idxs = jr.permutation(rng_key, idxs)

    def get_batch(idx, idxs=idxs):
        start_idx = idx * batch_size
        step_size = jnp.minimum(n - start_idx, batch_size)
        ret_idx = lax.dynamic_slice_in_dim(idxs, idx * batch_size, step_size)
        batch = {
            name: lax.index_take(array, (ret_idx,), axes=(0,))
            for name, array in zip(data._fields, data)
        }
        return batch

    return DataLoader(num_batches, idxs, get_batch)
