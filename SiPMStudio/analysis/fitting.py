import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from SiPMStudio.processing.functions import sipm_fall
from SiPMStudio.processing.functions import sipm_rise


def super_pulse(waves_data):
    return np.mean(waves_data, axis=0)


def fit_super_pulse(time_series, wave, display=False):
    peaks = find_peaks(wave, distance=1000, width=4)
    peak_location = peaks[0][0]
    rise_region = wave.iloc[0:peak_location]
    fall_region = wave.iloc[peak_location:]
    rise_guess = [1, 1, 1, 1]
    fall_guess = [1, 1, 1, 1]
    coeffs_1, cov_1 = curve_fit(sipm_rise, time_series[0:peak_location], rise_region)
    coeffs_2, cov_2 = curve_fit(sipm_fall, time_series[peak_location:], fall_region)
    if display:
        t_rise = np.linspace(0, time_series[peak_location], 1000)
        t_fall = np.linspace(time_series[peak_location], time_series[-1], 1000)
        sns.set_style("whitegrid")
        plt.figure()
        plt.plot(time_series, wave)
        plt.plot(t_rise, sipm_rise(t_rise, coeffs_1[0], coeffs_1[1], coeffs_1[2]), color="red")
        plt.plot(t_fall, sipm_fall(t_fall, coeffs_2[0], coeffs_2[1], coeffs_2[2]), color="red")
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.legend(["Super-Pulse", "fit"])
    return coeffs_1, coeffs_2





