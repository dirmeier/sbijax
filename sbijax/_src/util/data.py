from jax import numpy as jnp, tree_flatten


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.vstack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def stack_data(data, also_data):
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
    stacked =  tree_stack([data, also_data])
    return stacked

