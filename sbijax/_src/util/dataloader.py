from __future__ import annotations

from typing import TYPE_CHECKING

import grain
import jax
import jax.flatten_util
import jax.tree_util
import numpy as np
from jax import numpy as jnp
from jax import random as jr

if TYPE_CHECKING:
  from collections.abc import Iterator

  from jax import Array

  from sbijax._src.util.types import PyTree


# pylint: disable=too-few-public-methods
class DataLoader:
  """Grain-backed batches of a columnar data set.

  The loader shuffles and batches row indices and gathers each batch from the
  columns, so no per-row objects are built. When a seed is given, the data is
  reshuffled every epoch.

  Attributes:
      num_samples: number of rows the batches cover per epoch.
  """

  def __init__(
    self,
    columns: dict[str, np.ndarray],
    batch_size: int,
    *,
    drop_remainder: bool = False,
    seed: int | None = None,
  ):
    """Constructs a data loader.

    Args:
        columns: the data set as named arrays that share their leading axis
        batch_size: size of each batch
        drop_remainder: drop the last batch if it has fewer than
            ``batch_size`` rows
        seed: the shuffling seed, or ``None`` to keep the order

    Raises:
        ValueError: if ``batch_size`` is smaller than one or the columns have
            different numbers of rows
    """
    if batch_size < 1:
      raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    lengths = {key: len(column) for key, column in columns.items()}
    if len(set(lengths.values())) > 1:
      raise ValueError(f"columns have different lengths: {lengths}")
    self._columns = columns
    self._length = next(iter(lengths.values()), 0)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._seed = seed
    self._epoch = 0
    self.num_samples = self._length
    if drop_remainder:
      self.num_samples = self._length // batch_size * batch_size

  def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
    """Iterates over the data set in a new order each epoch.

    Yields:
        a dictionary of batched numpy arrays
    """
    indices = grain.MapDataset.range(self._length)
    if self._seed is not None:
      indices = indices.shuffle(seed=self._seed + self._epoch)
      self._epoch += 1
    indices = indices.batch(
      self._batch_size, drop_remainder=self._drop_remainder
    )
    batches = indices.map(
      lambda idx: {key: col[idx] for key, col in self._columns.items()}
    )
    yield from batches.to_iter_dataset()


def as_batch_iterators(
  rng_key: Array,
  data: PyTree,
  batch_size: int,
  split: float,
  shuffle: bool,
) -> tuple[DataLoader, DataLoader]:
  """Creates two grain-backed batch iterators from a data set.

  Args:
      rng_key: a jax random key
      data: a named tuple with elements 'y' and 'theta' all data
      batch_size: size of each batch
      split: fraction of data to use for training data set. Rest is used
          for validation data set.
      shuffle: shuffle the data set or no

  Returns:
      a training and a validation batch iterator
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
    max(1, min(batch_size, n_train)),
    shuffle,
    drop_remainder=True,
  )
  val_itr = as_batch_iterator(val_rng_key, y_val, batch_size, shuffle)

  return train_itr, val_itr


def as_batch_iterator(
  rng_key: Array,
  data: PyTree,
  batch_size: int,
  shuffle: bool,
  drop_remainder: bool = False,
) -> DataLoader:
  """Creates a grain-backed batch iterator from a data set.

  Args:
      rng_key: a jax random key
      data: a named tuple with elements 'y' and 'theta' all data
      batch_size: size of each batch
      shuffle: shuffle the data set or no
      drop_remainder: drop the last batch if it has fewer than
          ``batch_size`` elements

  Returns:
      a grain-backed batch iterator
  """
  flat_theta = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(
    data["theta"]
  )
  columns = {"y": np.asarray(data["y"]), "theta": np.asarray(flat_theta)}
  return DataLoader(
    columns,
    batch_size,
    drop_remainder=drop_remainder,
    seed=_grain_seed(rng_key) if shuffle else None,
  )


def _grain_seed(rng_key: Array) -> int:
  # grain takes an integer seed, not a jax key
  max_int32 = jnp.iinfo(jnp.int32).max
  return int(jr.randint(rng_key, shape=(), minval=0, maxval=max_int32))
