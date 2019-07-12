import numpy as np
import seaborn as sns
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