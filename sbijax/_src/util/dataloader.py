import grain
import jax.tree_util
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax._src.flatten_util import ravel_pytree

from sbijax._src.util.types import PyTree


# pylint: disable=too-few-public-methods
class DataLoader:
  """Batches of a data set, reshuffled every epoch when a seed is given.

  Args:
      data: the unbatched data set
      num_samples: number of elements the batches cover per epoch
      batch_size: size of each batch
      drop_remainder: drop the last batch if it has fewer than
          ``batch_size`` elements
      seed: the shuffling seed, or ``None`` to keep the order
  """

  def __init__(
    self, data, num_samples, batch_size, *, drop_remainder=False, seed=None
  ):
    self._data = data
    self.num_samples = num_samples
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._seed = seed
    self._epoch = 0

  def __iter__(self):
    """Iterate over the data set in a new order each epoch."""
    data = self._data
    if self._seed is not None:
      data = data.shuffle(seed=self._seed + self._epoch)
      self._epoch += 1
    data = data.batch(self._batch_size, drop_remainder=self._drop_remainder)
    yield from data.to_iter_dataset()


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

  # an incomplete training batch can be smaller than the contrastive or
  # atomic sets some losses draw from it, so it is dropped; a training set
  # smaller than one batch forms a single batch
  train_itr = as_batch_iterator(
    train_rng_key,
    y_train,
    min(batch_size, n_train),
    shuffle,
    drop_remainder=True,
  )
  val_itr = as_batch_iterator(val_rng_key, y_val, batch_size, shuffle)

  return train_itr, val_itr


# pylint: disable=missing-function-docstring
def as_batch_iterator(
  rng_key: Array, data: PyTree, batch_size, shuffle, drop_remainder=False
):
  """Create a data batch iterator from a data set.

  Args:
      rng_key: a jax random key
      data: a named tuple with elements 'y' and 'theta' all data
      batch_size: size of each batch
      shuffle: shuffle the data set or no
      drop_remainder: drop the last batch if it has fewer than
          ``batch_size`` elements

  Returns:
      a tensorflow iterator
  """
  n = data["y"].shape[0]
  data = [
    {"y": y, "theta": theta}
    for y, theta in zip(
      data["y"],
      jax.vmap(lambda x: ravel_pytree(x)[0])(data["theta"]),
      strict=False,
    )
  ]
  itr = grain.MapDataset.source(data)
  return as_batched_numpy_iterator(
    rng_key, itr, n, batch_size, shuffle, drop_remainder
  )


# ruff: noqa: PLR0913
def as_batched_numpy_iterator(
  rng_key: Array,
  data: grain.MapDataset,
  iter_size,
  batch_size,
  shuffle,
  drop_remainder=False,
):
  """Create a data batch iterator from a tensorflow data set.

  Args:
      rng_key: a jax random key
      data: a named tuple with elements 'y' and 'theta' all data
      iter_size: total number of elements in the data set
      batch_size: size of each batch
      shuffle: shuffle the data set or no
      drop_remainder: drop the last batch if it has fewer than
          ``batch_size`` elements

  Returns:
      a tensorflow iterator
  """
  seed = None
  if shuffle:
    # grain takes an integer seed, not a jax key
    max_int32 = jnp.iinfo(jnp.int32).max
    seed = int(jr.randint(rng_key, shape=(), minval=0, maxval=max_int32))
  if drop_remainder:
    iter_size = iter_size // batch_size * batch_size
  return DataLoader(
    data, iter_size, batch_size, drop_remainder=drop_remainder, seed=seed
  )
