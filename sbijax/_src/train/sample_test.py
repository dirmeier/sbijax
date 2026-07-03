from sbijax._src.train._types import ObjectiveFns, TrainFns
from sbijax._src.train.sample import sample


def test_sample_dispatches_to_sample_fn():
  def sample_fn(rng, params, y, *, sampler=None, **kw):
    return ("drew", sampler, kw)

  obj = ObjectiveFns(TrainFns(None, None, None), sample_fn, extra=None)
  out = sample(0, obj, {"p": 1}, "y", sampler="S", n_samples=7)
  assert out == ("drew", "S", {"n_samples": 7})
