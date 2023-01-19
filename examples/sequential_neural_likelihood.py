import distrax
import haiku as hk
import optax

from jax import numpy as jnp
from surjectors.bijectors.masked_coupling import MaskedCoupling
from surjectors.conditioners.mlp import mlp_conditioner
from surjectors.distributions.transformed_distribution import \
    TransformedDistribution
from surjectors.surjectors.chain import Chain
from surjectors.util import make_alternating_binary_mask

from sbi import SNL
from jax import random


def _bijector_fn(params):
    means, log_scales = jnp.split(params, 2, -1)
    return distrax.ScalarAffine(means, jnp.exp(log_scales))


def _conditioner_fn(ndim_hidden_layers, n_hidden_layers, output_dim):
    return mlp_conditioner(
        [ndim_hidden_layers] * n_hidden_layers + [output_dim],
    )


def make_model(dim):
    def _flow(method, **kwargs):
        layers = []
        for i in range(2):
            mask = make_alternating_binary_mask(dim, i % 2 == 0)
            output_dim = dim * 2
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=_conditioner_fn(16, 2, output_dim=output_dim),
            )
            layers.append(layer)
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(dim), jnp.ones(dim)),
            reinterpreted_batch_ndims=1,
        )
        td = TransformedDistribution(
            base_distribution,
            chain
        )
        return td(method, **kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


def prior_model_fns():
    p = distrax.Uniform(jnp.full(2, -3.0), jnp.full(2, 3.0))
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p_noise = distrax.Normal(jnp.zeros_like(theta), 1.0)
    noise = p_noise.sample(seed=seed)
    return theta + 0.1 * noise


model = make_model(2)
snl = SNL((prior_model_fns, simulator_fn), model)

optimizer = optax.adam(1e-4)
snl.train(random.PRNGKey(23), jnp.full(2, 2), optimizer)
