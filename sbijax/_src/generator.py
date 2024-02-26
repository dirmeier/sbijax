from collections import namedtuple

import tensorflow as tf
from jax import Array
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
    rng_key: Array, data: named_dataset, batch_size, split, shuffle
):
    """Create two data batch iterators from a data set.

    Args:
        rng_key: a jax random key
        data: a named tuple with elements 'y' and 'theta' all data
        batch_size: size of each batch
        split: fraction of data to use for training data set. Rest is used
            for validation data set.
        shuffle: shuffle the data set or no

    Returns:
        returns two iterators
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
def as_batch_iterator(rng_key: Array, data: named_dataset, batch_size, shuffle):
    """Create a data batch iterator from a data set.

    Args:
        rng_key: a jax random key
        data: a named tuple with elements 'y' and 'theta' all data
        batch_size: size of each batch
        shuffle: shuffle the data set or no

    Returns:
        a tensorflow iterator
    """
    # hack, cause the tf stuff doesn't support jax keys :)
    max_int32 = jnp.iinfo(jnp.int32).max
    seed = jr.randint(rng_key, shape=(), minval=0, maxval=max_int32)
    itr = tf.data.Dataset.from_tensor_slices(data)
    itr = (
        itr.shuffle(
            10 * batch_size,
            seed=int(seed),
            reshuffle_each_iteration=shuffle,
        )
        .batch(batch_size)
        .prefetch(buffer_size=batch_size)
        .as_numpy_iterator()
    )
    return itr
