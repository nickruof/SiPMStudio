import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from scipy.sparse import diags
from scipy.stats import expon
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from SiPMStudio.processing.functions import multi_gauss
from SiPMStudio.plots.plots import plot_delay_times
from SiPMStudio.io.file_settings import read_file

warnings.filterwarnings("ignore", "PeakPropertyWarning: some peaks have a width of 0")


def collect_files(path, data_dir="UNFILTERED"):
    dirs_array = []
    file_array = []
    for dirpath, dirnames, filenames in os.walk(path):
        if dirnames:
            dirs_array.append(dirnames)
    runs = []
    waves = []

    for name in dirs_array[0]:
        if name.startswith("runs_"):
            runs.append(name)
        elif name.startswith("waves_"):
            waves.append(name)

    for run in runs:
        data_path = path+run+"/"+data_dir
        files = os.listdir(data_path)
        file_targets = []
        os.chdir(data_path)
        for file in files:
            if os.path.getsize(data_path+"/"+file) > 50:
                runs.append(data_path+"/"+file)
            else:
                pass
    for wave in waves:
        data_path = path+wave+"/"+data_dir
        files = os.listdir(data_path)
        file_targets = []
        os.chdir(data_path)
        for file in files:
            if os.path.getsize(data_path+"/"+file) > 50:
                waves.append(data_path+"/"+file)
            else:
                pass

    return runs, waves


def time_interval(params_data, waves_data=None):
    interval = params_data["TIMETAG"].iloc[-1] - params_data["TIMETAG"].iloc[0]
    interval = interval * 1.0e-12
    return interval


def spectrum_peaks(params_data, waves_data=None, n_bins=2000, hist_range=None, min_dist=0.0, min_height=0.0, width=4.0, display=False):
    peaks = []
    bin_edges = []
    if display:
        plt.figure()
        [bin_vals, bin_edges, _patches] = plt.hist(params_data, bins=n_bins, range=hist_range, density=True, edgecolor="none")
        bin_width = bin_edges[1] - bin_edges[0]
        peaks, _properties = find_peaks(bin_vals, height=min_height, distance=min_dist/bin_width, width=width)
        print(str(len(peaks)) + " peaks found!")
        bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
        plt.plot(bin_centers[peaks], bin_vals[peaks], ".r")
        plt.yscale("log")
        plt.show()
    else:
        [bin_vals, bins_edges, _patches] = plt.hist(params_data, bins=n_bins, range=hist_range, density=True, edgecolor="none")
        bin_width = bin_edges[1] - bin_edges[0]
        peaks, _properties = find_peaks(bin_vals, height=min_height, distance=min_dist/bin_width, width=width)

    x_values = bin_edges[:-1]
    return x_values[peaks]


def fit_multi_gauss(params_data, waves_data=None, min_dist=0.0, min_height=0.0, params=None, display=False):
    bins = np.linspace(start=0, stop=max(params_data), num=int(max(params_data)))
    bin_vals, bin_edges = np.histogram(params_data, bins=bins, density=True)
    fit = []
    popt = []
    if params is None:
        peaks, _properties = find_peaks(bin_vals, height=min_height, distance=min_dist, width=width)
        if display:
            plt.figure()
            plt.bar(bin_edges[:-1], bin_vals, width=1000, edgecolor="none")
            plt.plot(peaks, bin_vals[peaks],".r")
            plt.yscale("log")
            plt.show()
        amplitudes = bin_vals[peaks]
        sigmas = [17]*len(peaks)  # method needed to avoid hard coded sigma guess
        guess = []
        for i, peak in enumerate(peaks):
            guess.append(peak)
            guess.append(amplitudes[i])
            guess.append(sigmas[i])
        popt, pcov = curve_fit(multi_gauss, xdata=bins[:-1], ydata=bin_vals, p0=guess)
        fit = multi_gauss(bins[:-1], *popt)
    else:
        fit = multi_gauss(bins[:-1], *params)
    if display:
        plt.figure()
        plt.bar(bin_edges[:-1], bin_vals, edgecolor="none")
        plt.plot(bins[:-1], fit, color="red")
        plt.yscale("log")
        plt.show()
    return popt


def gain(digitizer, path, file_name, sipm, sum_len=1, settings_option="peaks", params_data=None, waves_data=None):
    diffs = []
    gain_average = 1
    peaks = np.array(read_file(path, file_name, file_type="runs")[settings_option])
    for i in range(len(peaks)-1):
        diffs.append(peaks[i+1]-peaks[i])
    gain_average = sum(diffs[0:sum_len]) / float(len(diffs[0:sum_len]))
    sipm.gain.append(gain_average)
    gain_magnitude = gain_average * digitizer.e_cal/1.6e-19
    sipm.gain_magnitude.append(gain_magnitude)    
    return gain_average, gain_magnitude


def dark_count_rate(path, file_name, sipm, settings_option="wave_peaks", bounds=None, params_data=None, waves_data=None, display=False):
    rate = []
    all_times = []
    settings = read_file(path, file_name, file_type="waves")[settings_option]
    for i, wave in waves_data.iterrows():
        peaks, _properties = find_peaks(x=wave, height=settings["min_height"],
                                        distance=settings["min_dist"], width=settings["width"])
        rate.append(len(peaks) / (len(wave) * 2e-9))
        if len(peaks) > 0:
            times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
            all_times = np.append(all_times, [times])

    # pulse_rate
    average_pulse_rate = sum(rate) / len(rate)
    sipm.pulse_rate.append(average_pulse_rate)

    # exponential fit to delay time histogram
    M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
    all_dts = M_diag @ all_times
    all_dts = np.delete(all_dts, -1)
    if bounds is None:
        bounds = [0, 1e5]
    dts_fit = all_dts[(all_dts > bounds[0]) & (all_dts < bounds[1])]
    exp_fit = expon.fit(dts_fit)
    sipm.dcr_fit.append(1/(exp_fit[1]*1e-9))

    if display:
        plt.figure()
        plot_delay_times(dts=all_dts, fit=True)
        plt.show()
        plt.close()

    return average_pulse_rate, 1/(exp_fit[1]*1e-9)


def excess_charge_factor(sipm, params_data=None, waves_data=None):
    return np.divide(sipm.pulse_rate, sipm.dcr_fit)


def cross_talk(path, file_name, sipm, settings_option="peaks", params_data=None, waves_data=None):
    peaks = np.array(read_file(path, file_name, file_type="runs")[settings_option])
    energy_data = params_data["E_SHORT"]
    counts = pd.Series(data=[1]*energy_data.shape[0])
    half_pe = peaks[0] - sipm.gain[-1]/2
    one_half_pe = peaks[1] - sipm.gain[-1]/2
    # bins = list(range(int(max(params_data["E_SHORT"]))))
    # bin_vals, _bin_edges = np.histogram(params_data["E_SHORT"], bins=bins)
    total_counts1 = counts[energy_data > half_pe].sum()
    total_counts2 = counts[energy_data > one_half_pe].sum()
    prob = total_counts2 / total_counts1
    sipm.cross_talk.append(prob)
    return prob


def delay_times(params_data, waves_data, min_height, min_dist, width=0):
    all_times = []

    for i, wave in enumerate(waves_data.values):
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        if len(peaks) > 0:
            times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
            all_times = np.append(all_times, [times])
    M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
    all_dts = M_diag @ all_times
    all_dts = np.delete(all_dts, -1)
    return all_dts


def heights(wave_data, min_height, min_dist, width=0):
    all_heights = []

    for i, wave in wave_data.iterrows():
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        for height in wave.values[peaks]:
            all_heights.append(height)
    all_heights = np.array(all_heights)
    all_heights = np.delete(all_heights, -1)
    return all_heights


def delay_time_vs_height(params_data, wave_data, min_height, min_dist, width=0):
    all_dts = []
    all_heights = []
    all_times = []

    for i, wave in wave_data.iterrows():
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
        peak_heights = wave_data.iloc[i, :].values[peaks]
        all_times = np.append(all_times, [times])
        all_heights = np.append(all_heights, [peak_heights])
    if len(all_times) == 0 or len(all_heights) == 0:
        print("No peaks found!")
        return all_dts, all_heights
    else:
        M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
        all_dts = M_diag @ all_times
        all_dts = np.delete(all_dts, -1)
        all_heights = np.delete(all_heights, -1)
        return all_dts, all_heights
