from SiPMStudio.plots import plots_base


def plot_iv(ax, voltage, current):
    plots_base.interp_plot(ax, voltage, current, n_points=100)
    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("Current (A)")
