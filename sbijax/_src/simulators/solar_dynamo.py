import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.scipy.special import erf
from scipy.signal.windows import hann
from tensorflow_probability.substrates.jax import distributions as tfd

__all__ = ["solar_dynamo"]


def _sample_timeseries(
    seed, y0, alpha_min, alpha_max, epsilon_max, len_timeseries=200
):
    a = tfd.Uniform(alpha_min, alpha_max).sample(
        seed=seed, sample_shape=(len_timeseries,)
    )
    noise = tfd.Uniform(0.0, epsilon_max).sample(
        seed=seed, sample_shape=(len_timeseries,)
    )

    def _fn(fs, arrays):
        alpha, epsilon = arrays
        f, pn = fs
        f = _babcock_leighton_fn(pn)
        pn = _babcock_leighton(pn, alpha, epsilon)
        return (f, pn), (f, pn)

    _, (f, y) = lax.scan(_fn, (y0, y0), (a, noise))
    return f.T, y.T, a.T, noise.T


def _babcock_leighton_fn(p, b_1=0.6, w_1=0.2, b_2=1.0, w_2=0.8):
    f = 0.5 * (1.0 + erf((p - b_1) / w_1)) * (1.0 - erf((p - b_2) / w_2))
    return f


def _babcock_leighton(p, alpha, epsilon):
    p = alpha * _babcock_leighton_fn(p) * p + epsilon
    return p


def _simulate(seed, theta):
    orig_shape = theta.shape
    if theta.ndim == 2:
        theta = theta[None, :, :]

    alpha_min = theta[..., 0]
    alpha_max = alpha_min + theta[..., 1]
    epsilon_max = theta[..., 2]
    y0 = jnp.ones(theta.shape[:-1])

    _, y, _, _ = _sample_timeseries(
        seed, y0, alpha_min, alpha_max, epsilon_max, 100
    )

    y = jnp.swapaxes(y, 1, 0)
    if len(orig_shape) == 2:
        y = y.reshape((*orig_shape[:1], 100))
    return y


# ruff: noqa: PLR0913, E501
def solar_dynamo(summarize_data=False):
    """Solar dynamo model.

    Constructs prior and simulator functions

    Returns:
      returns a tuple of three objects. The first is a
      tfd.JointDistributionNamed serving as a prior distribution. The second
      is a simulator function that can be used to generate data. The third
      is None (since the likelihood is intractable and to be consistent with other
      models).

    References:
      Albert, Carlo, et al., Learning summary statistics for Bayesian inference with autoencoders, 2022
    """

    def prior_fn():
        return tfd.JointDistributionNamed(
            dict(
                theta=tfd.Independent(
                    tfd.Uniform(
                        jnp.array([0.9, 0.05, 0.02]),
                        jnp.array([1.4, 0.25, 0.15]),
                    ),
                    reinterpreted_batch_ndims=1,
                )
            )
        )

    def summarize(ys):
        window = hann(ys.shape[1])
        window = window.reshape(1, -1)
        fourier_range = np.arange(0, ys.shape[1], 6)

        fs = np.fft.ifft(window * ys, axis=1)
        ss = np.abs(fs[:, fourier_range])
        return fourier_range, ss

    def simulator(seed, theta):
        theta = theta["theta"].reshape(-1, 3)
        ys = _simulate(seed, theta)
        if summarize_data:
            _, summ = summarize(ys)
            return summ
        return ys

    return prior_fn(), simulator, None
