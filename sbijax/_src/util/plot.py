import arviz as az
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator


def plot_trace(inference_data):
    var_sizes = inference_data.posterior.sizes
    max_dim = np.max(
        [v for k, v in dict(var_sizes).items() if k not in ["chain", "draw"]]
    )
    colors = sns.blend_palette(["#636363", "#b26679"], n_colors=max_dim)
    with az.style.context(["arviz-white"], after_reset=True):
        plt.rcParams["font.family"] = "Times New Roman"
        axes = az.plot_trace(
            inference_data,
            compact_prop={"color": colors},
        )
        for ax in axes.flatten():
            ax.spines[["right", "top"]].set_visible(False)
            ax.spines.left.set_linewidth(0.5)
            ax.spines.bottom.set_linewidth(0.5)
            ax.yaxis.set_major_locator(AutoLocator())
            ax.title.set_fontsize(15)
            ax.yaxis.set_tick_params(labelsize="large")
            ax.xaxis.set_tick_params(labelsize="large")
            ax.xaxis.set_tick_params(width=0.5, length=2)
            ax.yaxis.set_tick_params(width=0.5, length=2)
            ax.grid(which="major", axis="both", alpha=0.5)
    plt.tight_layout()
    return axes


def plot_posterior(inference_data):
    colors = sns.blend_palette(["#636363", "#b26679"], n_colors=2)
    with az.style.context(["arviz-white"], after_reset=True):
        plt.rcParams["font.family"] = "Times New Roman"
        axes = az.plot_posterior(
            inference_data,
            color=colors[1],
            kind="hist",
            hdi_prob=0.9,
            edgecolor="black",
        )
        for ax in axes.flatten():
            ax.spines[["right", "top"]].set_visible(False)
            ax.spines.left.set_linewidth(0.5)
            ax.spines.bottom.set_linewidth(0.5)
            ax.xaxis.set_tick_params(width=0.5, length=2)
            ax.yaxis.set_tick_params(width=0.5, length=2)
            ax.grid(which="major", axis="both", alpha=0.25)
    plt.tight_layout()
    return axes
