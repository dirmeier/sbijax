import random
from typing import Callable, NamedTuple

import distrax
import haiku as hk
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


class slice:
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
            n_thin=0,
            max_width=jnp.inf,
        ):
            def body_fn(current_state, i):
                order_key, key = random.split(random.fold_in(rng_key, i))

                n = current_state.n[0]
                print(n)
                positions = current_state.position["theta"]
                widths = current_state.widths["theta"]

                order = random.choice(
                    order_key,
                    jnp.arange(len(positions)),
                    shape=(len(positions),),
                    replace=False,
                )

                def inner_body_fn(carry, idx):
                    position, width = carry
                    xi, wi = _sample_conditionally(
                        key,
                        idx,
                        position,
                        logdensity_fn,
                        width[idx],
                        max_width=max_width,
                    )
                    position = position.at[idx].set(xi)
                    nw = width[idx] + (wi - width[idx]) / (n + 1)
                    width = width.at[idx].set(nw)

                    return (position, width), (position, width)

                (new_positions, new_widths), _ = jax.lax.scan(
                    inner_body_fn, (positions, widths), order
                )
                new_state = {"theta": new_positions}
                new_widths = {"theta": new_widths}
                new_state = SliceState(
                    new_state,
                    jnp.atleast_1d(logdensity_fn(new_state)),
                    new_widths,
                    jnp.atleast_1d(n + 1.0),
                )
                return new_state, new_state

            new_state, _ = jax.lax.scan(body_fn, state, jnp.arange(n_thin + 1))
            return new_state

        return one_step

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        prior: distrax.Distribution,
        *,
        n_thin=0,
    ) -> MCMCSamplingAlgorithm:
        step = cls._kernel()

        def init_fn(position: PyTree):
            return cls._init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return step(rng_key, state, logdensity_fn, n_thin=n_thin)

        return MCMCSamplingAlgorithm(init_fn, step_fn)


def _sample_conditionally(rng, idx, x, logdensity_fn, wi, max_width):
    def cond_lp_fn(x, t):
        x = x.at[idx].set(t)
        return logdensity_fn({"theta": x})

    key1, key2, key3, key4 = random.split(rng, 4)

    xci = x[idx]
    logu = cond_lp_fn(x, idx) + jnp.log(1.0 - random.uniform(key1))
    lx = xci - wi * random.uniform(key2)
    ux = lx + wi

    # while cond_lp_fn(x, lx) >= logu and xci - lx < max_width:
    #     lx -= wi
    # while cond_lp_fn(x, ux) >= logu and ux - xci < max_width:
    #     ux += wi
    #
    # xi = (ux - lx) * random.uniform(key3) + lx
    #
    # while cond_lp_fn(x, xi) < logu:
    #     if xi < xci:
    #         lx = xi
    #     else:
    #         ux = xi
    #     xi = (ux - lx) * rng.rand() + lx
    xi = (ux - lx) * random.uniform(key4) + lx
    return xi, ux - lx


prior = distrax.Independent(distrax.Normal(jnp.zeros(2), 1.0), 1)


observed = distrax.Normal(jnp.array([10, -10]), 1.0).sample(
    seed=2, sample_shape=(10,)
)


def logdensity_fn(theta, observed=observed):
    logpdf = distrax.Normal(theta, 1.0)
    return jnp.sum(logpdf.log_prob(observed))


def logdensity(x):
    return logdensity_fn(**x)


ss = slice(logdensity, prior, n_thin=123433)

state = ss.init({"theta": prior.sample(seed=22)})

asdss = ss.step(random.PRNGKey(1), state)

print(asdss)
