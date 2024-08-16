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
# # Examples
#
# Here we implement the models given in the `Examples` section.

# %%
import arviz as az
import jax
import numpy as np
import optax
import sbijax
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoLocator, MaxNLocator
from jax import numpy as jnp, random as jr
from jax._src.flatten_util import ravel_pytree
from tensorflow_probability.substrates.jax import distributions as tfd


# %% [markdown]
# We implement a custom function to visualize posterior pairs.

# %%
def plot_posteriors(obj):
    _, axes = plt.subplots(figsize=(12, 10), nrows=5, ncols=5)
    with az.style.context(["arviz-doc"], after_reset=True):
        for i in range(0, 5):
            for j in range(0, 5):
                ax = axes[i, j]
                if i < j:
                    ax.axis('off')
                else:
                    ax.hexbin(obj[..., j], obj[..., i], gridsize=50, bins='log', cmap='viridis')
                ax.spines.left.set_linewidth(.5)
                ax.spines.bottom.set_linewidth(.5)
                ax.spines.right.set_linewidth(.5)
                ax.spines.top.set_linewidth(.5)
                ax.xaxis.set_major_locator(MaxNLocator(2))
                ax.yaxis.set_major_locator(MaxNLocator(2))
                ax.xaxis.set_tick_params(width=1, length=2, labelsize=25)
                ax.yaxis.set_tick_params(width=1, length=2, labelsize=25)
                if i < 4:
                    ax.set_xticklabels([])
                    ax.xaxis.set_tick_params(width=0., length=0)
                if j != 0:
                    ax.set_yticklabels([])
                    ax.yaxis.set_tick_params(width=0., length=0)
                ax.grid(which='major', axis='both', alpha=0.5)
        for i in range(5):
            axes[i, i].hist(obj[..., i], color="black")
    return axes


# %%
def plot_ess_and_trace(inference_results):
    _, axes = plt.subplots(figsize=(10, 8), nrows=5, ncols=3)

    with az.style.context(["arviz-doc", "arviz-viridish"], after_reset=True):
        plt.rcParams["font.family"] = "Times New Roman"
        colors = sns.color_palette("ch:s=-.2,r=.6", as_cmap=False, n_colors=10, desat=.8)
        ax = az.plot_ess(
            inference_results,
            ax=[axes[i, 2] for i in range(5)],
            color="#36454F",
            extra_kwargs={"color": list(colors[-5])},
            kind="evolution"
        )
        for ax in [axes[i, 2] for i in range(5)]:
            ax.axline((0, 0), slope=1, color="black", ls="--")
        colors = sns.color_palette("mako_r", as_cmap=False, n_colors=10)
        az.plot_rank(inference_results, ax=[axes[i, 1] for i in range(5)],  kind='vlines', colors=colors, vlines_kwargs={"alpha":0.15}, marker_vlines_kwargs={"linestyle":'None', "marker": "o", "ms":3, "alpha": 0.75})
        for i in range(5):
            for j in range(10):
                axes[i, 0].plot(slice_samples.reshape(10, 5000, 5)[j, :, i], color=colors[j], alpha=0.15)
                axes[i, 0].set_ylabel(rf"$\theta_{i}$", fontsize=15)
                axes[i, 1].set_ylabel(None)
                axes[i, 2].set_ylabel(None)
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(None)
            ax.spines[['right', 'top']].set_visible(False)
            ax.spines.left.set_linewidth(.5)
            ax.spines.bottom.set_linewidth(.5)
            ax.yaxis.set_major_locator(AutoLocator())
            ax.set_xlabel(None)
            if i in [13, 14]:
                ax.set_xlabel("Total number of draws", fontsize=15)
            if i == 12:
                ax.set_xlabel("Number of draws per chain", fontsize=15)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            #ax.legend(bbox_to_anchor=(0.95, .7), frameon=False)
            ax.yaxis.set_tick_params(labelsize=12)
            ax.xaxis.set_tick_params(labelsize=12)
            ax.xaxis.set_tick_params(width=0.5, length=2)
            ax.yaxis.set_tick_params(width=0.5, length=2)
            ax.grid(which='major', axis='both', alpha=0.5)
            if i != [12,13,14]:
                ax.set_xticklabels([])
            if i in [2,5,8,11,14]:
                ax.set_yticklabels([])
            axes[4, 2].legend(["Bulk ESS", "Tail ESS"], fontsize=12)
        return axes


# %% [markdown]
# We then define the generative model.

# %%
def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))
    ), batch_ndims=0)
    return prior


def simulator_fn(seed, theta):
    theta = theta["theta"]
    theta = theta[:, None, :]
    us_key, noise_key = jr.split(seed)

    def _unpack_params(ps):
        m0 = ps[..., [0]]
        m1 = ps[..., [1]]
        s0 = ps[..., [2]] ** 2
        s1 = ps[..., [3]] ** 2
        r = jnp.tanh(ps[..., [4]])
        return m0, m1, s0, s1, r

    m0, m1, s0, s1, r = _unpack_params(theta)
    us = tfd.Normal(0.0, 1.0).sample(
        seed=us_key, sample_shape=(theta.shape[0], theta.shape[1], 4, 2)
    )
    xs = jnp.empty_like(us)
    xs = xs.at[:, :, :, 0].set(s0 * us[:, :, :, 0] + m0)
    y = xs.at[:, :, :, 1].set(
        s1 * (r * us[:, :, :, 0] + jnp.sqrt(1.0 - r**2) * us[:, :, :, 1]) + m1
    )    
    y = y.reshape((*theta.shape[:1], 8))    
    return y


# %%
y_obs = jnp.array([[
    -0.9707123,
    -2.9461224,
    -0.4494722,
    -3.4231849,
    -0.13285634,
    -3.364017,
    -0.85367596,
    -2.4271638,
]])

# %% [markdown]
# ### MCMC

# %%
from functools import partial
from jax import scipy as jsp
from sbijax import as_inference_data
from sbijax.mcmc import sample_with_nuts, sample_with_slice


# %%
def likelihood_fn(theta, y):
    mu = jnp.tile(theta[:2], 4)
    s1, s2 = theta[2] ** 2, theta[3] ** 2
    corr = s1 * s2 * jnp.tanh(theta[4])
    cov = jnp.array([[s1**2, corr], [corr, s2**2]])
    cov = jsp.linalg.block_diag(*[cov for _ in range(4)])
    p = tfd.MultivariateNormalFullCovariance(mu, cov)
    return p.log_prob(y)


def log_density_fn(theta, y):
    prior_lp = tfd.JointDistributionNamed(dict(
        theta=tfd.Uniform(jnp.full(5, -3.0), jnp.full(5, 3.0))
    )).log_prob(theta)
    likelihood_lp = likelihood_fn(theta, y)
    lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
    return lp


# %%
log_density = partial(log_density_fn, y=y_obs)

def lp(theta):
    return jax.vmap(log_density)(theta)

slice_samples = sample_with_slice(
    jr.PRNGKey(0),
    lp,
    prior_fn().sample,
    n_chains=10,
    n_samples=10_000,
    n_warmup=5_000
)

# %%
slice_inference_data = as_inference_data({"theta": slice_samples.reshape(10, 5000, 5)}, y_obs)

# %%
_, axes = plt.subplots(figsize=(9, 2), ncols=2)
sbijax.plot_rhat_and_ress(slice_inference_data, axes=axes)
plt.tight_layout()
plt.savefig("./figs/example-rhat_ress-slice.pdf", dpi=200)
plt.show()

    # %%
    plot_ess_and_trace(slice_inference_data)
plt.tight_layout()
plt.savefig("./figs/example-triple_plot-slice.pdf", dpi=200)
plt.show()

# %%
plot_posteriors(slice_samples.reshape(-1, 5))
plt.tight_layout()
plt.savefig("./figs/example-pair_plot-slice.pdf", dpi=200)
plt.show()

# %% [markdown]
# ### SNLE

# %%
from sbijax import SNLE, inference_data_as_dictionary
from sbijax.nn import make_maf

# %%
n_dim_data = 8
n_layer_dimensions, hidden_sizes = (8, 8, 5, 5, 5), (64, 64)
neural_network = make_maf(
    n_dim_data, 
    n_layer_dimensions=n_layer_dimensions,
    hidden_sizes=hidden_sizes
)

fns = prior_fn, simulator_fn
snle = SNLE(fns, neural_network)

# %%
data, snle_params = None, {}
for i in range(15):
    data, _ = snle.simulate_data_and_possibly_append(
        jr.fold_in(jr.PRNGKey(1), i),
        params=snle_params,
        observable=y_obs,
        data=data,
    )
    snle_params, info = snle.fit(
        jr.fold_in(jr.PRNGKey(2), i), data=data
    )

# %%
snle_inference_results, diagnostics = snle.sample_posterior(
    jr.PRNGKey(5), snle_params, y_obs, n_samples=5_000, n_warmup=2_500, n_chains=10
)

# %%
plot_posteriors(
    inference_data_as_dictionary(snle_inference_results.posterior)["theta"],
)
plt.tight_layout()
plt.savefig("./figs/example-pair_plot-snle.pdf", dpi=200)
plt.show()

# %% [markdown]
# ### FMPE

# %%
from sbijax import FMPE
from sbijax.nn import make_cnf

# %%
n_dim_theta = 5
n_layers, hidden_size = 5, 128
neural_network = make_cnf(n_dim_theta, n_layers, hidden_size)

fns = prior_fn, simulator_fn
fmpe = FMPE(fns, neural_network)

# %%
data, _ = fmpe.simulate_data(
    jr.PRNGKey(1),
    n_simulations=20_000,
)
fmpe_params, info = fmpe.fit(
    jr.PRNGKey(2),
    data=data,
    optimizer=optax.adam(0.001),
    n_early_stopping_delta=0.00001,
    n_early_stopping_patience=30
)

# %%
fmpe_inference_results, diagnostics = fmpe.sample_posterior(
    jr.PRNGKey(5), fmpe_params, y_obs, n_samples=25_000
)

# %%
plot_posteriors(
    inference_data_as_dictionary(fmpe_inference_results.posterior)["theta"],
)
plt.tight_layout()
plt.savefig("./figs/example-pair_plot-fmpe.pdf", dpi=200)
plt.show()

# %% [markdown]
# ### SMC-ABC

# %%
from sbijax import NASS, SMCABC, inference_data_as_dictionary
from sbijax.nn import make_nass_net

# %%
n_embedding_dim, hidden_sizes = 5, (64, 64)
neural_network = make_nass_net(n_embedding_dim, hidden_sizes)

fns = prior_fn, simulator_fn
model_nass = NASS(fns, neural_network)

data, _ = model_nass.simulate_data(jr.PRNGKey(1), n_simulations=20_000)
params_nass, _ = model_nass.fit(jr.PRNGKey(2), data=data, n_early_stopping_patience=25)


# %%
def summary_fn(y):
    s = model_nass.summarize(params_nass, y)
    return s

def distance_fn(y_simulated, y_observed):
    diff = y_simulated - y_observed
    dist = jax.vmap(lambda el: jnp.linalg.norm(el))(diff)
    return dist


# %%
model_smc = SMCABC(fns, summary_fn, distance_fn)

smc_inference_results, _ = model_smc.sample_posterior(
    jr.PRNGKey(5),
    y_obs,
    n_rounds=10,
    n_particles=10_000,
    eps_step=0.825,
    ess_min=2_000
)

# %%
plot_posteriors(
    inference_data_as_dictionary(smc_inference_results.posterior)["theta"],
)
plt.tight_layout()
plt.savefig("./figs/example-pair_plot-smc.pdf", dpi=200)
plt.show()

# %% [markdown]
# ## Session info

# %%
import session_info

session_info.show(html=False)
