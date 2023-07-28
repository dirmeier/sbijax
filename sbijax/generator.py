from collections import namedtuple

import chex
from jax import lax
from jax import numpy as jnp
from jax import random

named_dataset = namedtuple("named_dataset", "y theta")


# pylint: disable=missing-class-docstring,too-few-public-methods
class DataLoader:
    def __init__(self, num_batches, idxs, get_batch):
        self.num_batches = num_batches
        self.idxs = idxs
        self.get_batch = get_batch

    def __call__(self, idx, idxs=None):
        if idxs is None:
            idxs = self.idxs
        return self.get_batch(idx, idxs)


# pylint: disable=missing-function-docstring
def as_batch_iterators(
    rng_key: chex.PRNGKey, data: named_dataset, batch_size, split, shuffle
):
    n = data.y.shape[0]
    n_train = int(n * split)
    if shuffle:
        data = named_dataset(
            *[
                random.permutation(rng_key, el, independent=False)
                for _, el in enumerate(data)
            ]
        )
    y_train = named_dataset(*[el[:n_train, :] for el in data])
    y_val = named_dataset(*[el[n_train:, :] for el in data])
    train_rng_key, val_rng_key = random.split(rng_key)

    train_itr = as_batch_iterator(train_rng_key, y_train, batch_size, shuffle)
    val_itr = as_batch_iterator(val_rng_key, y_val, batch_size, shuffle)

    return train_itr, val_itr


# pylint: disable=missing-function-docstring
def as_batch_iterator(
    rng_key: chex.PRNGKey, data: named_dataset, batch_size, shuffle
):
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
        idxs = random.permutation(rng_key, idxs)

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
