from SiPMStudio.core.data_loading import Keithley2450

import seaborn as sns


def iv_curve(scope, data):
    scope.load_data(data)
    return scope.voltage, scope.current


def plot_iv(current, voltage):
    sns.set_style("whitegrid")
    plt.figure()
    plt.rc('text', usetex=True)
    plt.scatter(voltage, current)
    plt.plot(voltage, current)
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Current (Micro Amps)")
