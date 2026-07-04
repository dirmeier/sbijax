import jax.numpy as jnp

from sbijax._src.train._types import (
  Info,
  ObjectiveFns,
  SummaryFns,
  TrainFns,
  TrainingState,
  next_round,
)


def test_records_and_next_round():
  ts = TrainingState(params={"w": jnp.zeros(2)}, opt_state=())
  tf = TrainFns(
    init_fn=lambda *a: None, step_fn=lambda *a: None, eval_fn=lambda *a: None
  )
  obj = ObjectiveFns(train=tf, sample_fn=lambda *a: None, extra=None)
  summ = SummaryFns(train=tf, summarize_fn=lambda *a: None)
  assert ts.params["w"].shape == (2,) and obj.train is tf and summ.train is tf
  assert (
    next_round(None) == 0
    and next_round(Info(round=0, losses=jnp.zeros((3, 2)))) == 1
  )
