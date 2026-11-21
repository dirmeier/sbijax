import grain
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random as jr

from sbijax._src.util.dataloader import DataLoader, as_batch_iterators


def _columns(n):
  return {
    "y": np.arange(2 * n, dtype=np.float32).reshape(n, 2),
    "theta": np.arange(n, dtype=np.float32).reshape(n, 1),
  }


def test_unshuffled_batches_cover_rows_in_order():
  loader = DataLoader(_columns(10), 4)
  batches = list(loader)
  assert [b["y"].shape[0] for b in batches] == [4, 4, 2]
  np.testing.assert_array_equal(
    np.concatenate([b["theta"] for b in batches]), _columns(10)["theta"]
  )
  assert loader.num_samples == 10


def test_drop_remainder_trims_batches_and_num_samples():
  loader = DataLoader(_columns(10), 4, drop_remainder=True)
  assert [b["y"].shape[0] for b in loader] == [4, 4]
  assert loader.num_samples == 8


def test_shuffled_batches_match_legacy_per_row_pipeline():
  # pins that a fixed seed yields the same batches as the old loader
  cols = _columns(10)
  rows = [{"y": cols["y"][i], "theta": cols["theta"][i]} for i in range(10)]
  legacy = grain.MapDataset.source(rows).shuffle(seed=7).batch(4)
  for old, new in zip(
    legacy.to_iter_dataset(), DataLoader(cols, 4, seed=7), strict=True
  ):
    np.testing.assert_array_equal(old["y"], new["y"])
    np.testing.assert_array_equal(old["theta"], new["theta"])


def test_reshuffles_each_epoch_and_is_reproducible():
  a = DataLoader(_columns(50), 50, seed=3)
  b = DataLoader(_columns(50), 50, seed=3)
  first, second = next(iter(a))["theta"], next(iter(a))["theta"]
  assert not np.array_equal(first, second)
  np.testing.assert_array_equal(first, next(iter(b))["theta"])


def test_empty_dataset_yields_no_batches():
  loader = DataLoader(_columns(0), 4, seed=1)
  assert list(loader) == [] and loader.num_samples == 0
  data = {"y": jnp.ones((10, 1)), "theta": {"a": jnp.zeros((10, 2))}}
  _, val_itr = as_batch_iterators(jr.key(0), data, 4, 1.0, True)
  assert list(val_itr) == [] and val_itr.num_samples == 0


def test_mismatched_column_lengths_raise():
  with pytest.raises(ValueError):
    DataLoader({"y": np.zeros((3, 1)), "theta": np.zeros((4, 1))}, 2)


def test_non_positive_batch_size_raises():
  with pytest.raises(ValueError):
    DataLoader(_columns(3), 0)


def test_as_batch_iterators_flattens_theta_and_splits():
  data = {
    "y": jnp.ones((100, 3)),
    "theta": {"a": jnp.zeros((100,)), "b": jnp.zeros((100, 2))},
  }
  train_itr, val_itr = as_batch_iterators(jr.key(0), data, 32, 0.9, True)
  assert train_itr.num_samples == 64 and val_itr.num_samples == 10
  batch = next(iter(train_itr))
  assert isinstance(batch["theta"], np.ndarray)
  assert batch["theta"].shape == (32, 3)
