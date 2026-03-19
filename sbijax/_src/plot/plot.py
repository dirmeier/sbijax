import arviz as az
import arviz_plots
import jax
import numpy as np
import xarray
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator


def plot_trace(inference_data: xarray.DataTree):
  """MCMC trace plot.

  Args:
      inference_data: an inference data object received from calling
          `sample_posterior` of an SBI algorithm
      axes: an array of matplotlib axes
      **kwargs: additional parameters passed to Arviz

  Returns:
      the same array of matplotlib axes with added plots
  """
  pl = az.plot_trace(inference_data)
  return pl


def plot_posterior(inference_data: xarray.DataTree):
  """Posterior histogram plot.

  Args:
      inference_data: an inference data object received from calling
          `sample_posterior` of an SBI algorithm
      axes: an array of matplotlib axes

  Returns:
      the same array of matplotlib axes with added plots
  """
  pl = arviz_plots.plot_dist(inference_data)
  return pl


def plot_loss_profile(
  losses: jax.Array, axes: pyplot.Axes = None
) -> pyplot.Axes:
  """Visualize the training and validation loss profile.

  Args:
      losses: a jax.Array of training and validation losses
      axes: a matplotlib axes

  Returns:
      the same array of matplotlib axes with added plots
  """
  if axes is None:
    _, axes = pyplot.subplots(figsize=(5, 3), sharey=False, sharex=False)
  axes.plot(losses[:, 0], label="Training loss", linestyle=(0, (3, 1, 1, 1)))
  axes.plot(losses[:, 1], label="Validation loss", linestyle=(0, (5, 1)))
  axes.yaxis.set_major_locator(MaxNLocator(5))
  axes.legend()
  return axes


def plot_rank(inference_data: xarray.DataTree):
  """Rank statistics plots.

  Args:
      inference_data: an inference data object received from calling
          `sample_posterior` of an SBI algorithm

  Returns:
      the same array of matplotlib axes with added plots
  """
  pl = az.plot_rank(inference_data)
  return pl


# ruff: noqa: PLR2004
def plot_ess(inference_data: xarray.DataTree):
  """Effective sample size plot.

  Args:
      inference_data: an inference data object received from calling
          `sample_posterior` of an SBI algorithm

  Returns:
      the same array of matplotlib axes with added plots
  """
  pl = az.plot_ess(inference_data)
  return pl


def plot_rhat_and_ress(
  inference_data: xarray.DataTree, axes: np.ndarray[pyplot.Axes] = None
) -> np.ndarray[pyplot.Axes]:
  r"""Split-$\hat{R}$ and relative effective sample size plot.

  Args:
      inference_data: an inference data object received from calling
          `sample_posterior` of an SBI algorithm
      axes: an array of matplotlib axes

  Returns:
      the same array of matplotlib axes with added plots
  """
  rhats = az.rhat(inference_data)
  rhats = np.concatenate([np.array(v) for k, v in rhats.data_vars.items()])
  rhats = np.squeeze(rhats)
  ress = az.ess(inference_data, relative=True)
  ress = np.concatenate([np.array(v) for k, v in ress.data_vars.items()])
  ress = np.squeeze(ress)

  if axes is None:
    _, axes = pyplot.subplots(ncols=2)
  axes[0].plot(
    rhats, range(len(rhats)), marker="o", linestyle="None", color="black"
  )
  axes[0].hlines(range(len(rhats)), np.ones(len(rhats)), rhats, color="black")
  axes[0].axvline(1.05, color="darkgrey", alpha=0.5, linestyle="dashed")
  axes[0].axvline(1.1, color="darkgrey", alpha=0.5, linestyle="dashed")
  axes[0].axvline(1.0, color="black", alpha=0.5)
  axes[0].set_yticks(list(range(len(rhats))))

  if np.any(rhats < 1.0):
    axes[0].set_xlim(0.95)
  else:
    axes[0].set_xlim(0.99)
  if np.any(rhats >= 1.3):
    axes[0].axvline(1.3, color="dimgrey", alpha=0.5, linestyle="dashed")
    axes[0].set_xticks([1.0, 1.05, 1.1, 1.3])
  else:
    axes[0].set_xticks([1.0, 1.05, 1.1])
  axes[0].set_yticklabels([])
  axes[0].set_ylabel(r"$\theta$")
  axes[0].set_xlabel(r"Split-$\hat{R}$")

  axes[1].plot(
    ress, range(len(ress)), marker="o", linestyle="None", color="black"
  )
  axes[1].hlines(range(len(ress)), np.zeros(len(ress)), ress, color="black")
  axes[1].axvline(0.0, color="black", alpha=0.5)
  axes[1].axvline(0.1, color="darkgrey", alpha=0.5, linestyle="dashed")
  axes[1].axvline(0.5, color="darkgrey", alpha=0.5, linestyle="dashed")
  axes[1].axvline(1.0, color="darkgrey", alpha=0.5, linestyle="dashed")
  axes[1].set_xticks([0.0, 0.1, 0.5, 1.0])
  axes[1].set_yticks(list(range(len(ress))))
  axes[1].set_yticklabels([])
  axes[1].set_xlabel(r"Relative ESS")
  for i, ax in enumerate(axes):
    if i > 0:
      ax.set_ylabel(None)
  return axes
