import jax.tree_util
import tensorflow as tf
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.util.types import PyTree


# pylint: disable=missing-class-docstring,too-few-public-methods
class DataLoader:
    # noqa: D101
    def __init__(self, itr, num_samples):  # noqa: D107
        self._itr = itr
        self.num_samples = num_samples

    def __iter__(self):
        """Iterate over the data set."""
        yield from self._itr.as_numpy_iterator()


# pylint: disable=missing-function-docstring
def as_batch_iterators(
    rng_key: Array, data: PyTree, batch_size, split, shuffle
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
    n = data["y"].shape[0]
    n_train = int(n * split)

    if shuffle:
        idxs = jr.permutation(rng_key, jnp.arange(n))
        data = jax.tree_util.tree_map(lambda x: x[idxs], data)

    y_train = jax.tree_util.tree_map(lambda x: x[:n_train], data)
    y_val = jax.tree_util.tree_map(lambda x: x[n_train:], data)

    train_rng_key, val_rng_key = jr.split(rng_key)

    train_itr = as_batch_iterator(train_rng_key, y_train, batch_size, shuffle)
    val_itr = as_batch_iterator(val_rng_key, y_val, batch_size, shuffle)

    return train_itr, val_itr


def as_batched_numpy_iterator_from_tf(
    rng_key: Array, data: tf.data.Dataset, iter_size, batch_size, shuffle
):
    """Create a data batch iterator from a tensorflow data set.

    Args:
        rng_key: a jax random key
        data: a named tuple with elements 'y' and 'theta' all data
        iter_size: total number of elements in the data set
        batch_size: size of each batch
        shuffle: shuffle the data set or no

    Returns:
        a tensorflow iterator
    """
    # hack, cause the tf stuff doesn't support jax keys :)
    max_int32 = jnp.iinfo(jnp.int32).max
    seed = jr.randint(rng_key, shape=(), minval=0, maxval=max_int32)

    data = (
        data.shuffle(
            10 * batch_size,
            seed=int(seed),
            reshuffle_each_iteration=shuffle,
        )
        .batch(batch_size)
        .prefetch(buffer_size=batch_size)
    )
    return DataLoader(data, iter_size)


# pylint: disable=missing-function-docstring
def as_batch_iterator(rng_key: Array, data: PyTree, batch_size, shuffle):
    """Create a data batch iterator from a data set.

    Args:
        rng_key: a jax random key
        data: a named tuple with elements 'y' and 'theta' all data
        batch_size: size of each batch
        shuffle: shuffle the data set or no

    Returns:
        a tensorflow iterator
    """
    data = {
        "y": data["y"],
        "theta": jax.vmap(lambda x: ravel_pytree(x)[0])(data["theta"]),
    }
    itr = tf.data.Dataset.from_tensor_slices(data)
    return as_batched_numpy_iterator_from_tf(
        rng_key, itr, data["y"].shape[0], batch_size, shuffle
    )


def as_numpy_iterator_from_slices(data: PyTree, batch_size):
    itr = tf.data.Dataset.from_tensor_slices(data)
    itr = itr.batch(batch_size).prefetch(buffer_size=batch_size)
    itr = itr.as_numpy_iterator()
    return itr
