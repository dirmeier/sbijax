from jax import numpy as jnp
from jax.scipy.signal import welch
from jrnmm import simulate as simulate_jrnmm
from tensorflow_probability.substrates.jax import distributions as tfd


# ruff: noqa: PLR0913, E501
def jansen_rit(summarize_data=False):
  """Stochastic Jansen-Rit neural mass model.

  Constructs prior and simulator functions.

  Args:
    summarize_data: if true returns the data from the simulator in a summarized
      version of 5 values. Otherwise, returns the infection counts of the ODE.

  Returns:
    returns a tuple of three objects. The first is a
    tfd.JointDistributionNamed serving as a prior distribution. The second
    is a simulator function that can be used to generate data. The third
    is None (since the likelihood is intractable and to be consistent with other
    models).

  References:
    Ableidinger, Marko, et al., A stochastic version of the Jansen and Rit neural mass model: Analysis and numerics, 2017
  """

  def prior_fn():
    prior = tfd.JointDistributionNamed(
      dict(
        theta=tfd.Independent(
          tfd.Uniform(
            jnp.array([10.0, 50.0, 100.0, -20.0]),
            jnp.array([250.0, 500.0, 5000.0, 20.0]),
          ),
          1,
        )
      )
    )
    return prior

  def summarize(ys, fs=128.0, n_summaries=33):
    f, S = welch(ys, fs=fs, nperseg=2 * (n_summaries - 1), axis=1)
    return f, S

  def _simulate(seed, theta):
    Cs, mus, sigmas, gains = (
      theta[:, 0],
      theta[:, 1],
      theta[:, 2],
      theta[:, 3],
    )
    y = simulate_jrnmm(
      seed,
      dt=1 / 128,
      t_end=8.0 + 1.0 / 128.0,
      initial_states=jnp.array([0.08, 18, 15, -0.5, 0.0, 0.0]),
      Cs=Cs,
      mus=mus,
      sigmas=sigmas,
      gains=gains,
    )
    return y[:, :, 0]

  def simulator(seed, theta):
    theta = theta["theta"].reshape(-1, 4)
    ys = _simulate(seed, theta)
    if summarize_data:
      _, summ = summarize(ys)
      return summ
    return ys

  return prior_fn(), simulator, None
