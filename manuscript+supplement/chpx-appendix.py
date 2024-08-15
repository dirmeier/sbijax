# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: sbi-dev
#     language: python
#     name: sbi-dev
# ---

# %% [markdown]
# # Additional algorithm examples
#
# Here, we implement the additional examples from the appendix.

# %%
import haiku as hk
%matplotlib inline
import matplotlib.pyplot as plt
import sbijax

from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd


# %% [markdown]
# ## Generative model

# %% [markdown]
# We follow the example in the appendix and use the following generative model
#
# \begin{align}
# \theta &\sim \mathcal{N}_2(0, I)\\
# y \mid \theta &\sim 0.5 \ \mathcal{N}_2(\theta, I) + 0.5 \ \mathcal{N}_2(\theta, 0.01 I)
# \end{align}

# %%
def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Normal(jnp.zeros(2), 1)
    ), batch_ndims=0)
    return prior


def simulator_fn(seed, theta):
    mean = theta["theta"].reshape(-1, 2)
    n = mean.shape[0]
    data_key, cat_key = jr.split(seed)
    pi_categories = tfd.Categorical(logits=jnp.zeros(2))
    categories = pi_categories.sample(seed=cat_key, sample_shape=(n,))
    scales = jnp.array([1.0, 0.1])[categories].reshape(-1, 1)
    y = tfd.Normal(mean, scales).sample(seed=data_key)
    return y


# %% [markdown]
# ## Observation

# %%
y_observed = jnp.array([-1.0, 1.0])

# %% [markdown]
# ## NUTS

# %%
import arviz as az
from sbijax import as_inference_data
from sbijax.mcmc import sample_with_nuts
from functools import partial


# %%
def likelihood_fn(y, theta):
    mean = theta["theta"].reshape(1, 2)
    mix = tfd.Categorical(logits=jnp.zeros(2))
    likelihood = tfd.MixtureSameFamily(
        mixture_distribution=mix,
        components_distribution=tfd.MultivariateNormalDiag(
            jnp.concatenate([mean, mean], axis=0),
            jnp.concatenate([jnp.full((1, 2), 1.0), jnp.full((1, 2), 0.01)], axis=0),
        )
    )
    ll = likelihood.log_prob(y)
    return ll


# %%
def log_density_fn(theta, y):
    prior_lp = prior_fn().log_prob(theta)
    likelihood_lp = likelihood_fn(y, theta)
    lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
    return lp


# %%
log_density_partial = partial(log_density_fn, y=y_observed)
samples = sample_with_nuts(
    jr.PRNGKey(0),
    log_density_partial,
    prior_fn().sample
)
inference_results = as_inference_data(samples, jnp.squeeze(y_observed))

# %%
az.ess(inference_results, relative=True)

# %%
az.rhat(inference_results)

# %%
_, axes = plt.subplots(figsize=(6, 3), ncols=2)
sbijax.plot_posterior(inference_results, axes=axes)
plt.tight_layout()
plt.savefig("figs/appendix-mixture_model-nuts.pdf", dpi=200)
plt.show()

# %% [markdown]
# ## CMPE

# %%
import optax
from sbijax import CMPE
from sbijax._src.nn.make_consistency_model import ConsistencyModel


# %%
def make_model(dim):
    @hk.transform
    def _mlp(method, **kwargs):
        def _c_skip(time):
            return 1 / ((time - 0.001) ** 2 + 1)

        def _c_out(time):
            return 1.0 * (time - 0.001) / jnp.sqrt(1 + time**2)

        def _nn(theta, time, context, **kwargs):
            ins = jnp.concatenate([theta, time, context], axis=-1)
            outs = hk.nets.MLP([64, 64, dim])(ins)
            out_skip = _c_skip(time) * theta + _c_out(time) * outs
            return out_skip

        cm = ConsistencyModel(dim, _nn)
        return cm(method, **kwargs)

    return _mlp


# %%
fns = prior_fn, simulator_fn
model = CMPE(fns, make_model(2))

data, _ = model.simulate_data(jr.PRNGKey(1), n_simulations=10_000)
params, info = model.fit(
    jr.PRNGKey(2),
    data=data,
    optimizer=optax.adam(1e-3),
    n_early_stopping_patience=25
)

# %%
inference_results, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)

# %%
_, axes = plt.subplots(figsize=(6, 3), ncols=2)
sbijax.plot_posterior(inference_results, axes=axes)
plt.tight_layout()
plt.savefig("figs/appendix-mixture_model-cmpe.pdf", dpi=200)
plt.show()

# %% [markdown]
# ## NRE

# %%
from sbijax import NRE


# %%
def make_model():
    @hk.without_apply_rng
    @hk.transform
    def _mlp(inputs, **kwargs):
        return hk.nets.MLP([64, 64, 1])(inputs)

    return _mlp


# %%
fns = prior_fn, simulator_fn
model = NRE(fns, make_model())

data, _ = model.simulate_data(
    jr.PRNGKey(1), n_simulations=10000
)
params, info = model.fit(
    jr.PRNGKey(2),
    data=data,
    optimizer=optax.adam(1e-3),
    n_early_stopping_patience=25
)

# %%
inference_results, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)

# %%
_, axes = plt.subplots(figsize=(6, 3), ncols=2)
sbijax.plot_posterior(inference_results, axes=axes)
plt.tight_layout()
plt.savefig("figs/appendix-mixture_model-nre.pdf", dpi=200)
plt.show()

# %% [markdown]
# ## NPE

# %%
from sbijax import NPE
from surjectors import (
    Chain,
    MaskedAutoregressive,
    Permutation,
    ScalarAffine,
    TransformedDistribution,
)
from surjectors.nn import MADE
from surjectors.util import unstack


# %%
def make_flow(dim):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return ScalarAffine(means, jnp.exp(log_scales))

    def _flow(method, **kwargs):
        layers = []
        order = jnp.arange(dim)
        for i in range(5):
            layer = MaskedAutoregressive(
                bijector_fn=_bijector_fn,
                conditioner=MADE(
                    dim, [50, 50], 2,
                    w_init=hk.initializers.TruncatedNormal(0.001),
                    b_init=jnp.zeros,
                ),
            )
            order = order[::-1]
            layers.append(layer)
            layers.append(Permutation(order, 1))
        chain = Chain(layers)

        base_distribution = tfd.Independent(
            tfd.Normal(jnp.zeros(dim), jnp.ones(dim)),
            1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


# %%
fns = prior_fn, simulator_fn
model = NPE(fns, make_flow(2))

data, _ = model.simulate_data(
    jr.PRNGKey(1), n_simulations=10000
)
params, info = model.fit(
    jr.PRNGKey(2),
    data=data,
    n_iter=1000,
    optimizer=optax.adam(1e-3),
    n_early_stopping_patience=25
)

# %%
inference_results, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)

# %%
_, axes = plt.subplots(figsize=(6, 3), ncols=2)
sbijax.plot_posterior(inference_results, axes=axes)
plt.tight_layout()
plt.savefig("figs/appendix-mixture_model-npe.pdf", dpi=200)
plt.show()

# %% [markdown]
# ## NLE

# %%
from sbijax import NLE


# %%
def make_mdn(hidden_sizes=[50, 50], n_components=10, n_dimension=2):

    @hk.transform
    def mdn(method, y, x):
        n = x.shape[0]
        hidden = hk.nets.MLP([50, 50], activate_final=True)(x)
        logits = hk.Linear(n_components)(hidden)
        mu_sigma = hk.Linear(n_components * n_dimension * 2)(hidden)
        mu, sigma = jnp.split(mu_sigma, 2, axis=-1)

        mixture = tfd.MixtureSameFamily(
            tfd.Categorical(logits=logits),
                tfd.MultivariateNormalDiag(
                mu.reshape(n, n_components, n_dimension),
                    jnp.exp(sigma.reshape(n, n_components, n_dimension)),
            )
        )
        return mixture.log_prob(y)
    return mdn


# %%
fns = prior_fn, simulator_fn
model = NLE(fns, make_mdn())

data, _ = model.simulate_data(jr.PRNGKey(1))
params, info = model.fit(
    jr.PRNGKey(2),
    data=data,
    optimizer=optax.adam(1e-3),
    n_early_stopping_patience=25,
)

# %%
inference_results, _ = model.sample_posterior(jr.PRNGKey(3), params, y_observed)

# %%
_, axes = plt.subplots(figsize=(6, 3), ncols=2)
sbijax.plot_posterior(inference_results, axes=axes)
plt.tight_layout()
plt.savefig("figs/appendix-mixture_model-nle.pdf", dpi=200)
plt.show()

# %% [markdown]
# ## Session info

# %%
import session_info

session_info.show(html=False)
