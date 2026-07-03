import jax.numpy as jnp
from jax import random as jr

from sbijax._src.diagnostics.convergence import ess, mcmc_convergence, rhat


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


def test_mcmc_convergence_guards_on_chain_count():
  samples = {"theta": jr.normal(jr.PRNGKey(0), (4, 500, 2))}
  r, e = mcmc_convergence(samples, n_chains=4)
  assert r["theta"].shape == (2,) and e["theta"].shape == (2,)

  single = {"theta": jr.normal(jr.PRNGKey(0), (1, 500, 2))}
  r1, e1 = mcmc_convergence(single, n_chains=1)
  assert r1 is None and e1 is None
