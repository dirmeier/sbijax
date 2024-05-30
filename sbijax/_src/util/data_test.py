# pylint: skip-file

import chex
from jax import random as jr

from sbijax import NLE
from sbijax._src.nn.make_flow import make_maf
from sbijax._src.util.data import stack_data


def test_stack_data(prior_simulator_tuple):
    snl = NLE(prior_simulator_tuple, make_maf(2))
    n = 100
    data, _ = snl.simulate_data(jr.PRNGKey(1), n_simulations=n)
    also_data, _ = snl.simulate_data(jr.PRNGKey(2), n_simulations=n)
    stacked_data = stack_data(data, also_data)

    chex.assert_trees_all_equal(data["y"], stacked_data["y"][:n])
    chex.assert_trees_all_equal(data["theta"]["theta"], stacked_data["theta"]["theta"][:n])
    chex.assert_trees_all_equal(also_data["y"], stacked_data["y"][n:])
    chex.assert_trees_all_equal(also_data["theta"]["theta"], stacked_data["theta"]["theta"][n:])


def test_stack_data_with_none(prior_simulator_tuple):
    snl = NLE(prior_simulator_tuple, make_maf(2))
    n = 100
    data, _ = snl.simulate_data(jr.PRNGKey(1), n_simulations=n)
    stacked_data = stack_data(None, data)

    chex.assert_trees_all_equal(data["y"], stacked_data["y"])
    chex.assert_trees_all_equal(data["theta"]["theta"], stacked_data["theta"]["theta"])
