"""Per-family diagnostics records returned alongside posterior samples.

``Estimator.sample`` (and ``ABCSampler.sample``) return ``(samples, info)``
(DR-012). ``samples`` is the named prior pytree; ``info`` is one of these small
records. The conformance suite pins only the structural ``(pytree, record)``
pair, not a concrete type — MCMC methods report sampling diagnostics, amortized
methods report a trivial record.
"""

from typing import Any, NamedTuple

import jax


class MCMCSampleInfo(NamedTuple):
  """Sampling diagnostics for MCMC-based posteriors (NLE/NRE/SNLE).

  Attributes:
      acceptance_rate: mean post-warmup acceptance rate across chains and draws
      rhat: per-dimension split-R-hat pytree (same structure as ``samples``), or
          ``None`` when sampled with a single chain (R-hat is a between-chain
          statistic and undefined for one chain)
      ess: per-dimension effective-sample-size pytree, or ``None`` when sampled
          with a single chain
  """

  acceptance_rate: jax.Array
  rhat: Any = None
  ess: Any = None


class DirectSampleInfo(NamedTuple):
  """Sampling diagnostics for amortized posteriors (NPE/FMPE/CMPE).

  Attributes:
      n_samples: the number of posterior draws returned
  """

  n_samples: int
