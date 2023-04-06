import random
from typing import Callable, NamedTuple

import distrax
import jax
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
        widths = jax.tree_map(lambda x: jnp.full(x.shape, 0.01), position)
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
            positions = state.position["theta"]
            widths = state.widths["theta"]

            def inner_body_fn(carry, rn):
                seed, idx = rn
                positions, widths = carry
                xi, wi = _sample_conditionally(
                    seed, idx, positions, logdensity_fn, widths, n_doublings
                )
                positions = positions.at[idx].set(xi)
                nw = widths[idx] + (wi - widths[idx]) / (n + 1)
                widths = widths.at[idx].set(nw)
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

            new_state = {
                "theta": new_positions.reshape(state.position["theta"].shape)
            }
            new_widths = {
                "theta": new_widths.reshape(state.widths["theta"].shape)
            }
            new_state = SliceState(
                new_state,
                jnp.atleast_1d(logdensity_fn(new_state)),
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
    y = jnp.log(random.uniform(key)) + cond_lp_fn(x0)
    left, right = _doubling_fn(seed1, y, x0, cond_lp_fn, w0, n_doublings)
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
    return left, right


def _shrinkage_fn(seed, y, x0, cond_lp_fn, left, right, w):
    def cond_fn(state):
        x1, left, right, seed = state
        return jnp.where(
            jnp.logical_not(
                jnp.logical_and(
                    y < cond_lp_fn(x1),
                    _accept_fn(seed, y, x1, x0, cond_lp_fn, left, right, w),
                )
            ),
            True,
            False,
        )

    def body_fn(state):
        x1, left, right, seed = state
        key, seed = random.split(seed)
        left = jnp.where(x1 < x0, x1, left)
        right = jnp.where(x1 >= x0, x1, right)
        v = random.uniform(key)
        x1 = left + v * (right - left)
        return x1, left, right, seed

    key, seed = random.split(seed)
    v = random.uniform(key)
    x1 = left + v * (right - left)
    x1, left, right, seed = jax.lax.while_loop(
        cond_fn, body_fn, (x1, left, right, seed)
    )
    return x1


def _accept_fn(seed, y, x1, x0, cond_lp_fn, left, right, w):
    def cond_fn(state):
        x1, x0, left, right, w, D, is_acceptable = state
        ret = jnp.where(right - left > 1.1 * w, True, False)
        return ret

    def body_fn(state):
        x1, x0, left, right, w, D, is_acceptable = state
        mid = (left + right) / 2
        D = jnp.where(
            jnp.logical_or(
                jnp.logical_and(x0 < mid, x1 >= mid),
                jnp.logical_and(x0 >= mid, x1 < mid),
            ),
            True,
            D,
        )
        right = jnp.where(x1 < mid, mid, right)
        left = jnp.where(x1 < mid, left, mid)
        is_acceptable = jnp.where(
            jnp.logical_and(
                D,
                jnp.logical_and(y >= cond_lp_fn(left), y >= cond_lp_fn(right)),
            ),
            False,
            True,
        )
        return x1, x0, left, right, w, D, is_acceptable

    _, _, _, _, _, _, is_acceptable = jax.lax.while_loop(
        cond_fn, body_fn, (x1, x0, left, right, w, False, True)
    )
    return is_acceptable


#
# prior = distrax.Independent(distrax.Normal(jnp.zeros(4), 1.0), 1)
#
# observed1 = distrax.Normal(jnp.array([-4.0, 2.0]), 1.0).sample(
#     seed=3, sample_shape=(100,)
# )
# observed2 = distrax.Normal(jnp.array([-3.0, 0.0]), 1.0).sample(
#     seed=1, sample_shape=(100,)
# )
# observed = jnp.concatenate([observed2, observed1], axis=0)
#
#
# def logdensity_fn(theta, observed=observed):
#     lprio = prior.log_prob(theta)
#     logpdf1 = jnp.log(0.5) + distrax.Normal(theta[..., :2], 1.0).log_prob(observed)
#     logpdf2 = jnp.log(0.5) + distrax.Normal(theta[..., 2:], 1.0).log_prob(observed)
#     logpdf = logaddexp(logpdf2, logpdf1)
#     return jnp.sum(logpdf) + jnp.sum(lprio)
#
#
# def logdensity(x):
#     return logdensity_fn(**x)
#
# v = logdensity_fn(jnp.zeros(4))
#
#
# ss = slice(logdensity, prior, 5)
#
# n_chains = 4
# s = {"theta": prior.sample(seed=12, sample_shape=(n_chains, ))}
# state = jax.vmap(ss.init)(s)
# sampling_keys = jax.random.split(random.PRNGKey(3), 20000)
#
#
# def _step(states, rng_key):
#     keys = jax.random.split(rng_key, n_chains)
#     states = jax.vmap(ss.step)(keys, states)
#     return states, states
#
#
# _, states = jax.lax.scan(_step, state, sampling_keys)
# _ = states.position["theta"].block_until_ready()
#
#
# theta = states.position["theta"]
# theta = theta[10000:, :, :].reshape(-1, 4)
#
# import matplotlib.pyplot as plt
# plt.hist(theta[:, 0])
# plt.hist(theta[:, 1])
# plt.hist(theta[:, 2])
# plt.hist(theta[:, 3])
# plt.show()
#
# #
# # print(jnp.mean(theta[:, 0]))
# # print(jnp.mean(theta[:, 1]))
# # print(jnp.mean(theta[:, 2]))
# # print(jnp.mean(theta[:, 3]))
# #
# k = 2
