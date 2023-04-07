from typing import Callable, NamedTuple

import jax
import numpy as np
from blackjax.base import MCMCSamplingAlgorithm
from blackjax.types import PRNGKey, PyTree
from jax import numpy as jnp
from jax import random


class SliceState(NamedTuple):
    position: PyTree
    logdensity: jnp.ndarray
    widths: PyTree
    n: jnp.ndarray


class slice_sampler:
    @staticmethod
    def _init(position: PyTree, logdensity_fn: Callable):
        logdensity = logdensity_fn(position)
        widths = jax.tree_map(lambda x: jnp.full(x.shape, 1.0), position)
        return SliceState(
            position, jnp.atleast_1d(logdensity), widths, jnp.atleast_1d(0.0)
        )

    @staticmethod
    def _kernel():
        def one_step(
            rng_key: PRNGKey,
            state: SliceState,
            logdensity_fn: Callable,
            n_doublings=5,
        ):
            order_key, rng_key = random.split(rng_key)
            n = state.n[0]
            positions, unravel_fn = jax.flatten_util.ravel_pytree( state.position)
            widths, _ = jax.flatten_util.ravel_pytree(state.widths)

            def inner_body_fn(carry, rn):
                seed, idx = rn
                positions, widths = carry
                xi, wi = _sample_conditionally(
                    seed, idx, positions, logdensity_fn, widths, n_doublings
                )
                positions = positions.at[idx].set(xi)
                #nw = widths[idx] + (wi - widths[idx]) / (n + 1)
                #widths = widths.at[idx].set(nw)
                return (positions, widths), (positions, widths)

            if positions.ndim == 0:
                positions = jnp.atleast_1d(positions)
                widths = jnp.atleast_1d(widths)
                (new_positions, new_widths), _ = inner_body_fn(
                    (positions, widths), (rng_key, 0)
                )
            else:
                order = random.choice(
                    order_key,
                    jnp.arange(len(positions)),
                    shape=(len(positions),),
                    replace=False,
                )

                keys = random.split(rng_key, len(positions))
                (new_positions, new_widths), _ = jax.lax.scan(
                    inner_body_fn, (positions, widths), (keys, order)
                )

            new_positions = unravel_fn(new_positions)
            new_widths = unravel_fn(new_widths)
            new_state = SliceState(
                new_positions,
                jnp.atleast_1d(logdensity_fn(new_positions)),
                new_widths,
                jnp.atleast_1d(n + 1.0),
            )
            return new_state

        return one_step

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        n_doublings=1,
    ) -> MCMCSamplingAlgorithm:
        step = cls._kernel()

        def init_fn(position: PyTree):
            return cls._init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return step(rng_key, state, logdensity_fn, n_doublings=n_doublings)

        return MCMCSamplingAlgorithm(init_fn, step_fn)


def _sample_conditionally(
    seed, idx, position, logdensity_fn, widths, n_doublings
):
    def cond_lp_fn(xi_to_set):
        return logdensity_fn({"theta": position.at[idx].set(xi_to_set)})

    key, seed1, seed2 = random.split(seed, 3)
    x0, w0 = position[idx], widths[idx]
    y = cond_lp_fn(x0) - random.exponential(key)
    left, right, _ = _doubling_fn(seed1, y, x0, cond_lp_fn, w0, n_doublings)
    x1 = _shrinkage_fn(seed2, y, x0, cond_lp_fn, left, right, w0)
    return x1, right - left


def _doubling_fn(rng, y, x0, cond_lp_fn, w, n_doublings):
    """
    Implementation according to Fig 4 in [1]
    References
    -------
    [1] Radford Neil, Slice Sampling 2003
    """

    key1, key2, key3, key4 = random.split(rng, 4)
    left = x0 - w * random.uniform(key2)
    right = left + w
    K = n_doublings

    def cond_fn(state):
        left, right, K, _ = state
        return jnp.where(
            jnp.logical_and(
                K > 0,
                jnp.logical_or(y < cond_lp_fn(left), y < cond_lp_fn(right)),
            ),
            True,
            False,
        )

    def body_fn(state):
        left, right, K, seed = state
        rng, seed = random.split(seed)
        v = random.uniform(rng)
        left = jnp.where(v < 0.5, 2 * left - right, left)
        right = jnp.where(v < 0.5, right, 2 * right - left)
        return left, right, K - 1, seed

    left, right, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (left, right, K, rng)
    )
    return left, right, None


def _best_interval(x):
    k = x.shape[0]
    mults = jnp.arange(2 * k, k, -1, dtype=x.dtype)
    shifts = jnp.arange(k, dtype=x.dtype)
    indices = jnp.argmax(mults * x + shifts).astype(x.dtype)
    return indices


def _shrinkage_fn(seed, y, x0, cond_lp_fn, left, right, w):
    def cond_fn(state):
        *_, found = state
        return jnp.logical_not(found)

    def body_fn(state):
        x1, left, right, seed, _ = state
        key, seed = random.split(seed)
        v = random.uniform(key)
        x1 = left + v * (right - left)

        found = jnp.logical_and(
            y < cond_lp_fn(x1),
            _accept_fn(y, x1, x0, cond_lp_fn, left, right, w),
        )

        left = jnp.where(x1 < x0, x1, left)
        right = jnp.where(x1 >= x0, x1, right)

        return x1, left, right, seed, found

    x1, left, right, seed, _ = jax.lax.while_loop(
        cond_fn, body_fn, (x0, left, right, seed, False)
    )
    return x1


def _accept_fn(y, x1, x0, cond_lp_fn, left, right, w):
    def cond_fn(state):
        _, _, left, right, w, _, is_acceptable = state
        return jnp.logical_and(right - left > 1.1 * w, is_acceptable)

    def body_fn(state):
        x1, x0, left, right, w, D, _ = state
        mid = (left + right) / 2
        D = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_and(x0 < mid, x1 >= mid),
                jnp.logical_and(x0 >= mid, x1 < mid),
            ),
            D,
        )
        right = jnp.where(x1 < mid, mid, right)
        left = jnp.where(x1 >= mid, mid, left)

        left_is_not_acceptable = y >= cond_lp_fn(left)
        right_is_not_acceptable = y >= cond_lp_fn(right)
        interval_is_not_acceptable = jnp.logical_and(
            left_is_not_acceptable, right_is_not_acceptable
        )
        is_still_acceptable = jnp.logical_not(
            jnp.logical_and(D, interval_is_not_acceptable)
        )
        return x1, x0, left, right, w, D, is_still_acceptable

    *_, is_acceptable = jax.lax.while_loop(
        cond_fn, body_fn, (x1, x0, left, right, w, False, True)
    )
    return is_acceptable
