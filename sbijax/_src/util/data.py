import arviz as az
import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_flatten

from sbijax._src.util.types import PyTree


def _tree_stack(trees):
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.vstack(leave) for leave in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def stack_data(data: PyTree, also_data: PyTree) -> PyTree:
    """Stack two data sets.

    Args:
        data: one data set
        also_data: another data set

    Returns:
        returns the stack of the two data sets
    """
    if data is None:
        return also_data
    if also_data is None:
        return data
    stacked = _tree_stack([data, also_data])
    return stacked


def as_inference_data(samples: PyTree, observed: jax.Array) -> az.InferenceData:
    """Convert a PyTree to an inference data object.

    Args:
        samples: a PyTree of posterior samples
        observed: a jax.Array representing the observed data

    Returns:
        an inference data object
    """
    inf = az.InferenceData(
        posterior=az.dict_to_dataset(
            samples,
            coords={
                f"{k}_dim": np.arange(v.shape[-1]) for k, v in samples.items()
            },
            dims={k: [f"{k}_dim"] for k in samples.keys()},
        ),
        observed_data=az.dict_to_dataset({"y": observed}, default_dims=[]),
    )
    return inf


def inference_data_as_dictionary(posterior: az.InferenceData) -> PyTree:
    """Convert inference data to a PyTree.

    Args:
        posterior: the `posterior` variable of an inference data object

    Returns:
        a PyTree
    """
    posterior = posterior.to_dict()
    posterior = {
        k: jnp.array(v["data"]) for k, v in posterior["data_vars"].items()
    }
    posterior = {k: v.reshape(-1, v.shape[-1]) for k, v in posterior.items()}
    return posterior
