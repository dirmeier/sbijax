import haiku as hk
from jax import numpy as jnp
from jax import random as jr

from sbijax._src.experimental.nn.make_simformer import _SimFormer


def test_simformer_output_depends_on_time():
  @hk.transform
  def score(inputs, time, context):
    net = _SimFormer(mask=jnp.ones((4, 4)), n_heads=1, n_layers=1)
    return net(inputs, time, context, is_training=False)

  inputs, context = jnp.ones((3, 2)), jnp.ones((3, 2))
  params = score.init(jr.key(0), inputs, jnp.full(3, 0.1), context)
  early = score.apply(params, jr.key(1), inputs, jnp.full(3, 0.1), context)
  late = score.apply(params, jr.key(1), inputs, jnp.full(3, 0.9), context)
  assert not jnp.allclose(early, late)
