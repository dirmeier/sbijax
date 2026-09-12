from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax._src.train._types import ObjectiveFns, TrainFns
from sbijax._src.train.sample import sample


def _objective(samples):
  """An objective returning ``samples``, recording the kwargs it was given."""
  seen = {}

  def sample_fn(rng, params, y, *, sampler=None, **kw):
    seen.update(kw)
    return samples, ("info", sampler)

  return ObjectiveFns(TrainFns(None, None, None), sample_fn, extra=None), seen


def _nig_prior():
  return tfd.JointDistributionNamed(
    {
      "variance": tfd.InverseGamma(
        concentration=3.0 * jnp.ones(1), scale=2.0 * jnp.ones(1)
      ),
      "mean": lambda variance: tfd.Normal(jnp.zeros(2), jnp.sqrt(variance)),
    },
    batch_ndims=0,
  )


def test_sample_dispatches_to_sample_fn():
  obj, seen = _objective("drew")
  samples, info = sample(0, obj, {"p": 1}, "y", sampler="S", n_samples=7)
  assert samples == "drew"
  assert info == ("info", "S")
  assert seen == {"n_samples": 7}


def test_sample_does_not_forward_prior_to_sample_fn():
  # every amortized sample_fn ends in **kwargs and would swallow a forwarded
  # prior without effect, so the driver has to consume it instead
  obj, seen = _objective({"theta": jnp.zeros((1, 4, 3))})
  sample(0, obj, {}, "y", prior=_nig_prior(), n_samples=4)
  assert "prior" not in seen


def test_sample_names_flat_draws_after_the_prior():
  flat = jnp.arange(24.0).reshape(1, 8, 3)
  obj, _ = _objective({"theta": flat})
  samples, _ = sample(0, obj, {}, "y", prior=_nig_prior())
  assert samples["mean"].shape == (1, 8, 2)
  assert samples["variance"].shape == (1, 8, 1)


def test_sample_keeps_the_flat_layout_without_a_prior():
  obj, _ = _objective({"theta": jnp.zeros((1, 8, 3))})
  samples, _ = sample(0, obj, {}, "y")
  assert set(samples) == {"theta"}
