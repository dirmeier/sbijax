from collections import namedtuple

import arviz as az
import jax
import numpy as np
from jax import numpy as jnp


def as_inference_data(samples: dict[str, jax.Array], observed: jax.Array):
    inf = az.InferenceData(
        posterior=az.dict_to_dataset(
            samples,
            coords={f"{k}_dim": np.arange(v.shape[-1]) for k, v in samples.items()},
            dims={k: [f"{k}_dim"] for k in samples.keys()}
        ),
        observed_data=az.dict_to_dataset({"y": observed}, default_dims=[])
    )
    return inf


def mcmc_diagnostics(samples: az.InferenceData):
    return namedtuple(
        "MCMCDiagnostics", "rhat ess"
    )(az.rhat(samples), az.ess(samples))


def flatten(posterior):
    posterior = posterior.to_dict()
    posterior = {k: jnp.array(v["data"]) for k, v in posterior["data_vars"].items()}
    posterior = {k: v.reshape(-1, v.shape[-1]) for k, v in posterior.items()}
    return posterior
