import jax.numpy as jnp
from jax import random as jr

from sbijax._src.diagnostics.convergence import ess, rhat


def test_ess_and_rhat_on_named_pytree():
  # 4 chains, 500 draws, 2 dims of well-mixed standard normal noise
  samples = {
    "theta": jr.normal(jr.PRNGKey(0), (4, 500, 2)),
  }
  r = rhat(samples)
  e = ess(samples)
  assert r["theta"].shape == (2,)
  assert e["theta"].shape == (2,)
  assert jnp.all(r["theta"] < 1.1)
  assert jnp.all(e["theta"] > 0.0)
