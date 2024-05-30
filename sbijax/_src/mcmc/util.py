from collections import namedtuple



import arviz as az
import jax


def as_inference_data(samples: dict[str, jax.Array], observed: jax.Array):
    inf = az.InferenceData(
        posterior=az.dict_to_dataset(
            samples,
        ),
        observed_data=az.dict_to_dataset({"y": observed}, default_dims=[])
    )
    return inf


def mcmc_diagnostics(samples: az.InferenceData):
    return {"rhat": az.rhat(samples), "ess": az.ess(samples)}