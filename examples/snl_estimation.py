import distrax
import haiku as hk
import optax
from jax import numpy as jnp
from jax import random
from surjectors.bijectors.masked_coupling import MaskedCoupling
from surjectors.conditioners.mlp import mlp_conditioner
from surjectors.distributions.transformed_distribution import (
    TransformedDistribution,
)
from surjectors.surjectors.chain import Chain
from surjectors.util import make_alternating_binary_mask

from sbi import SNL


def prior_model_fns():
    p = distrax.Uniform(jnp.full(2, -3.0), jnp.full(2, 3.0))
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p_noise = distrax.Normal(jnp.zeros_like(theta), 1.0)
    noise = p_noise.sample(seed=seed)
    return theta + 0.1 * noise


def make_model(dim):
    def _bijector_fn(params):
        means, log_scales = jnp.split(params, 2, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(method, **kwargs):
        layers = []
        for i in range(2):
            mask = make_alternating_binary_mask(dim, i % 2 == 0)
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([8, 8, dim * 2]),
            )
            layers.append(layer)
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(dim), jnp.ones(dim)),
            reinterpreted_batch_ndims=1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
fns = ((prior_simulator_fn, prior_logdensity_fn), simulator_fn)
model = make_model(2)

snl = SNL(fns, model)

optimizer = optax.adam(1e-4)
params, all_losses, all_diagnostics = snl.train(
    random.PRNGKey(23), jnp.full(2, 2), optimizer
)
