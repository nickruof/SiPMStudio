from SiPMStudio.plots import plots_base

import seaborn as sns


def plot_iv(ax, voltage, current):
    sns.set_style("whitegrid")
    plots_base.line_plot(ax, voltage, current)
    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("Current (A)")
