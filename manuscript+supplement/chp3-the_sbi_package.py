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
# # The sbijax package
#
# `Sbijax` is a Python package for neural simulation-based inference and approximate Bayesian computation. Here we implement the code in chapter~3 of the manuscript.

# %%
import jax
import sbijax
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoLocator

# %% [markdown]
# ## 3.1 Model definition
#
# To do approximate inference using `sbijax`, a user first has to define a prior model and a simulator function which can be used to generate synthetic data. We will be using a simple bivariate Gaussian as an example with the following generative model:
#
# \begin{align}
# \mu &\sim \mathcal{N}_2(0, I)\\
# \sigma &\sim \mathcal{N}^+(1)\\
# y & \sim \mathcal{N}_2(\mu, \sigma^2 I)
# \end{align}
#
# Using TensorFlow Probability, the prior model and simulator are implemented like this:

# %%
from jax import numpy as jnp, random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        mean=tfd.Normal(jnp.zeros(2), 1.0),
        scale=tfd.HalfNormal(jnp.ones(1)),
    ), batch_ndims=0)
    return prior

def simulator_fn(seed: jr.PRNGKey, theta: dict[str, jax.Array]):
    p = tfd.Normal(jnp.zeros_like(theta["mean"]), 1.0)
    y = theta["mean"] + theta["scale"] * p.sample(seed=seed)
    return y


# %%
prior = prior_fn()
theta = prior.sample(seed=jr.PRNGKey(0), sample_shape=())
theta

# %%
prior.log_prob(theta)

# %%
simulator_fn(seed=jr.PRNGKey(1), theta=theta)

# %%
theta = prior.sample(seed=jr.PRNGKey(2), sample_shape=(2,))
theta

# %%
prior.log_prob(theta)

# %%
simulator_fn(seed=jr.PRNGKey(3), theta=theta)

# %% [markdown]
# ## 3.2 Algorithm definition
#
# Having defined a model of interest, i.e., the prior and simulator functions, we construct an inferential method.

# %%
import haiku as hk
import surjectors
import surjectors.nn
import surjectors.util
from sbijax import NLE

# %%
n_dim_data = 2
n_layers, hidden_sizes = 5, (64, 64)


# %%
def make_custom_affine_maf(n_dimension, n_layers, hidden_sizes):
    def _bijector_fn(params):
        means, log_scales = surjectors.util.unstack(params, -1)
        return surjectors.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(method, **kwargs):
        layers = []
        order = jnp.arange(n_dimension)
        for _ in range(5):
            layer = surjectors.MaskedAutoregressive(
                bijector_fn=_bijector_fn,
                conditioner=surjectors.nn.MADE(
                    n_dimension,
                    list(hidden_sizes),
                    2,
                    w_init=hk.initializers.TruncatedNormal(0.001),
                    b_init=jnp.zeros,
                ),
            )
            order = order[::-1]
            layers.append(layer)
            layers.append(surjectors.Permutation(order, 1))
        chain = surjectors.Chain(layers[:-1])
        base_distribution = tfd.Independent(
            tfd.Normal(jnp.zeros(n_dimension), jnp.ones(n_dimension)),
            1,
        )
        td = surjectors.TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


# %%
fns = prior_fn, simulator_fn
neural_network = make_custom_affine_maf(n_dim_data, n_layers, hidden_sizes)
model = NLE(fns, neural_network)

# %% [markdown]
# ## 3.3 Training and Inference
#
# Inference is then as easy as simulating some data, fitting the data to the model, a sampling from the approximate posterior. The data set is a dictionary of dictionaries (a PyTree in JAX lingo). It contains samples for the simulator function, called `y`, and parameter samples from the prior model, called `theta`.

# %%
data, _ = model.simulate_data(
    jr.PRNGKey(1),
    n_simulations=10_000,
)
data

# %% [markdown]
# We then fit the model using the typical flow matching loss.

# %%
params, losses = model.fit(
    jr.PRNGKey(2),
    data=data
)

# %% [markdown]
# Finally, we sample from the posterior distribution for a specific observation $y_{\text{obs}}$.

# %%
y_obs = jnp.array([-1.0, 1.0])
inference_results, diagnostics = model.sample_posterior(
    jr.PRNGKey(3), params, y_obs, n_chains=4, n_samples=10_000, n_warmup=5_000
)

# %%
print(inference_results)

# %%
print(inference_results.posterior)

# %% [markdown]
# ## 3.4 Model diagnostics and visualization
#
# `Sbijax` provides basic functionality to analyse posterior draws. We show some visualizations below.

# %%
_, axes = plt.subplots(figsize=(6, 2))
sbijax.plot_loss_profile(losses[1:], axes=axes)
plt.savefig("./figs/package-bivariate_model-loss_plot.pdf", dpi=200)
plt.show()

# %%
_, axes = plt.subplots(figsize=(9, 4), ncols=2, nrows=2)
sbijax.plot_trace(inference_results, axes=axes, compact_prop={'color': ['#636363', '#b26679']})
for i, ax in enumerate(axes.flatten()):
    if i % 2 == 0:
        ax.set_yticks([0, 0.5, 1, 1.5])
    elif i == 1:
        ax.set_yticks([-2, 0.0, 2.0])
    else:
        ax.set_yticks([0, 1, 2])
plt.savefig("./figs/package-trace_plot.pdf", dpi=200)
plt.tight_layout()
plt.show()

# %%
_, axes = plt.subplots(figsize=(9, 2), ncols=2)
sbijax.plot_rhat_and_ress(inference_results, axes)
plt.savefig("./figs/package-ress_rhat_plot.pdf", dpi=200)
plt.show()

# %%
_, axes = plt.subplots(figsize=(12, 5), ncols=3)
sbijax.plot_rank(inference_results, axes)
plt.tight_layout()
plt.savefig("./figs/package-rank_plot.pdf", dpi=200)
plt.show()

# %%
_, axes = plt.subplots(figsize=(12, 4), ncols=3)
sbijax.plot_ess(inference_results, axes)
for ax in axes.flatten():
    ax.set_yticks([0, 1000, 2000, 3000, 4000])
ax.legend(fontsize=14, bbox_to_anchor=(0.95, 0.7), frameon=False)
plt.tight_layout()
plt.savefig("./figs/package-ess_plot.pdf", dpi=200)
plt.show()

# %% [markdown]
# ## 3.5 Sequential inference
#
# `sbijax` supports sequential training.

# %%
from sbijax.util import stack_data

n_rounds = 1
data, params = None, {}
for i in range(n_rounds):
    new_data, _ = model.simulate_data(
        jr.fold_in(jr.PRNGKey(1), i),
        params=params,
        observable=y_obs,
        data=data,
    )
    data = stack_data(data, new_data)
    params, info = model.fit(jr.fold_in(jr.PRNGKey(2), i), data=data)

inference_results, diagnostics = model.sample_posterior(
    jr.PRNGKey(3), params, y_obs
)

# %% [markdown]
# ## Session info

# %%
import session_info

session_info.show(html=False)
