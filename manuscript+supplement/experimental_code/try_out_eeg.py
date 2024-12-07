import arviz as az
import jax
import numpy as np
import optax
import sbijax
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoLocator, MaxNLocator
from jax import numpy as jnp, random as jr
from jax._src.flatten_util import ravel_pytree
from tensorflow_probability.substrates.jax import distributions as tfd

import matplotlib.pyplot as plt
import numpy as np

import mne
import moabb

from jax.scipy.signal import welch
from moabb.datasets import Rodrigues2017



def jansen_rit_fn(len_timeseries=1025, t_end=8.0):
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    sdbmp = importr("sdbmsABC")

    rset_seed = robjects.r["set.seed"]
    rchol = robjects.r["chol"]
    rt = robjects.r["t"]

    A = 3.25
    B = 22.0
    a = 10.0
    b = 50.0
    vmax = 5.0
    v0 = 6.0
    r = 0.56
    sigma4 = 0.1
    sigma6 = 1

    burnin = 0
    h = 0.002
    grid = robjects.FloatVector(list(np.linspace(0, t_end, len_timeseries)))

    def fn(seed, theta):
        rset_seed(int(np.sqrt(seed[0])))
        C, mu, sigma, gain = theta.tolist()
        gain_abs = 10 ** (gain / 10)
        y0 = robjects.FloatVector(list(np.random.randn(6)))
        dm = sdbmp.exp_matJR(h, a, b)
        cm = rt(
            rchol(sdbmp.cov_matJR(h, robjects.FloatVector(
                [0, 0, 0, sigma4, sigma, sigma6]), a, b))
        )
        yt = gain_abs * jnp.array(
            sdbmp.Splitting_JRNMM_output_Cpp(
                h, y0, grid, dm, cm, mu, C, A, B, a, b, v0, r, vmax
            )
        )
        #yt = yt[int(burnin / h):]
        #yt = yt[::8]
        return yt - np.mean(yt)

    return fn


simulate_jansen_rit = jansen_rit_fn()

def prior_fn():
    prior = tfd.JointDistributionNamed(dict(
        C=tfd.Uniform(10.0, 250.0),
        mu=tfd.Uniform(50.0, 500.0),
        sigma=tfd.Uniform(100, 5000),
        gain=tfd.Uniform(-20, 20),
    ), batch_ndims=0)
    return prior



def simulator(seed, theta, len_timeseries=1025):
    Cs, mus, sigmas, gains = theta["C"], theta["mu"], theta["sigma"], theta["gain"]
    seeds = jr.split(seed, Cs.shape[0])

    ys = np.zeros((Cs.shape[0], len_timeseries))
    for i, (C, mu, sigma, gain, seed) in enumerate(zip(Cs, mus, sigmas, gains, seeds)):
        y = simulate_jansen_rit(seed, np.array([C, mu, sigma, gain]))
        ys[i] = np.array(y)
    return ys

def summarize(y, n_summaries=33, fs=128):
    _, summaries = welch(y, fs=fs, nperseg=2 * (n_summaries - 1), axis=1)
    summaries = 10 * np.log10(summaries)
    print(summaries.shape)
    return summaries


prior = prior_fn()

theta_synthetic = np.array([135, 220, 2000, 0])
y_synthetic = simulate_jansen_rit(jr.PRNGKey(0), theta_synthetic)

n = 100
theta_train = prior.sample(seed=jr.PRNGKey(2), sample_shape=(n,))
y_train = simulator(jr.PRNGKey(3), theta_train)

summaries_train = summarize(y_train)


from sbijax import FMPE, NLE, NPE
from sbijax.nn import make_cnf, make_maf

n_dim_data = 33
n_dim_theta = 4
n_layers, hidden_sizes = 5, (64, 64)
neural_network = make_maf(n_dim_theta, n_layers, hidden_sizes=hidden_sizes)

fns = prior_fn, None
estim = NPE(fns, neural_network)

data = {"y": summaries_train, "theta": theta_train}
params, info = estim.fit(
    jr.PRNGKey(1),
    data=data,
    optimizer=optax.adam(0.001),
    n_early_stopping_delta=0.00001,
    n_early_stopping_patience=30,
)
print(info)

posterior, diagnostics = estim.sample_posterior(
    jr.PRNGKey(2),
    params,
    observable=summaries_train[[0], :],
    n_samples=10_000,
)

az.plot_posterior(posterior)
az.plot_pair(posterior)
plt.show()