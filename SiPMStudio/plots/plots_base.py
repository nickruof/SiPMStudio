import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
sns.set_style("ticks")


def plot_hist(ax, values_array, bins=500, x_range=None, density=False, labels=None):
    colors = sns.color_palette()
    heights = []
    bin_edges = []
    all_patches = []
    for i, values in enumerate(values_array):
        [n, bins, patches] = ax.hist(values, bins=bins, range=x_range, edgecolor="none",
                                      color=colors[i], density=density, alpha=0.5)
        x_outline, y_outline = _hist_outline(n, bins)
        ax.plot(x_outline, y_outline, color=colors[i])
        heights.append(n)
        bin_edges.append(bins)
        all_patches.append(patches)
    if labels is not None:
        ax.legend(labels)
    return heights, bin_edges, all_patches


def _hist_outline(heights, bin_edges):
    x_values = []
    y_values = []

    for i in range(len(bin_edges)-1):
        if i == 0:
            x_values.append(bin_edges[i])
            y_values.append(0)
        x_values.append(bin_edges[i])
        x_values.append(bin_edges[i+1])
        x_values.append(bin_edges[i+1])
        if i == len(bin_edges)-2:
            y_values.append(heights[i])
            y_values.append(heights[i])
            y_values.append(0)
        else:
            y_values.append(heights[i])
            y_values.append(heights[i])
            y_values.append(heights[i+1])

    return np.array(x_values), np.array(y_values)


def line_plot(ax, x_values, y_values):
    ax.scatter(x_values, y_values)
    ax.plot(x_values, y_values)


def error_plot(ax, x_values, y_values):
    x_vals = [value.n for value in x_values]
    x_err = [value.s for value in x_values]
    y_vals = [value.n for value in y_values]
    y_err = [value.s for value in y_values]

    ax.scatter(x_vals, y_vals)
    ax.errorbar(x_vals, y_vals, y_err, x_err, capsize=1)


def interp_plot(ax, x_values, y_values, kind="cubic", n_points=None):
    if n_points is None:
        n_points = 100
    interp_func = interp1d(x_values, y_values, kind=kind)
    x_plot = np.linspace(min(x_values), max(x_values), n_points)
    y_plot = interp_func(x_plot)
    ax.plot(x_plot, y_plot)
