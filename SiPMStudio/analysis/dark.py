import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from scipy.sparse import diags
from scipy.stats import expon
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from SiPMStudio.processing.functions import multi_gauss
from SiPMStudio.processing.functions import multi_gauss_moyal


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


def spectrum_peaks(params_data, waves_data=None, min_dist=0.0, min_height=0.0, width=4.0, display=False):
    # bins = np.linspace(start=0, stop=max(params_data["E_SHORT"]), num=int(max(params_data["E_SHORT"])))
    bins = list(range(int(max(params_data["E_SHORT"]))))
    peaks = []
    if display:
        plt.figure()
        [bin_vals, bins_edges, _patches] = plt.hist(params_data["E_SHORT"], bins=bins, density=True, edgecolor="none")
        peaks, _properties = find_peaks(bin_vals, height=min_height, distance=min_dist, width=width)
        plt.plot(peaks, bin_vals[peaks], ".r")
        plt.yscale("log")
        plt.show()
    else:
        [bin_vals, bins_edges, _patches] = plt.hist(params_data["E_SHORT"], bins=bins, density=True, edgecolor="none")
        peaks, _properties = find_peaks(bin_vals, height=min_height, distance=min_dist, width=width)

    return peaks


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


def gain(digitizer, sipm, sum_len=1, params=None, peaks=None, params_data=None, waves_data=None):
    diffs = []
    gain_average = 1
    if peaks is None and params is not None:
        for i in range(0, len(params)-3, 3):
            diffs.append(params[i+3] - params[i])
    elif params is None and peaks is not None:
        for i in range(len(peaks)-1):
            diffs.append(peaks[i+1]-peaks[i])
    gain_average = sum(diffs[0:sum_len]) / float(len(diffs[0:sum_len]))
    sipm.gain.append(gain_average)
    gain_magnitude = gain_average * digitizer.e_cal/1.6e-19
    sipm.gain_magnitude.append(gain_magnitude)    
    return gain_average, gain_magnitude


def pe_units(df_data, gain, first_peak):
    data_array = df_data.values
    data_array = np.subtract(data_array, first_peak)
    data_array = np.divide(data_array, gain)
    data_array = np.add(data_array, 1)
    return pd.DataFrame(data=data_array, columns=df_data.columns)


def pulse_rate(sipm, min_height, min_dist, width=0, params_data=None, waves_data=None):
    rate = []
    for i, wave in waves_data.iterrows():
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        rate.append(len(peaks) / (len(wave)*2e-9))
    average_pulse_rate = sum(rate) / len(rate)
    sipm.pulse_rate.append(average_pulse_rate)
    return average_pulse_rate


def dcr_exp_fit(sipm, bounds=None, dts=None, params_data=None, waves_data=None):
    if bounds is None:
        bounds = [0, 1e5]
    dts_fit = dts[(dts > bounds[0]) & (dts < bounds[1])]
    exp_fit = expon.fit(dts_fit)
    sipm.dcr_fit.append(1/(exp_fit[1]*1e-9))
    return 1/(exp_fit[1]*1e-9)


def excess_charge_factor(sipm, params_data=None, waves_data=None):
    return np.divide(sipm.pulse_rate, sipm.dcr_fit)


def cross_talk(params_data, sipm, params=None, peaks=None, waves_data=None):
    if peaks is None and params is not None:
        index1 = int(params[0] - sipm.gain[-1]/2)
        index2 = int(params[3] - sipm.gain[-1]/2)
    elif params is None and peaks is not None:
        index1 = int(peaks[0] - sipm.gain[-1]/2)
        index2 = int(peaks[1] - sipm.gain[-1]/2)
    else:
        print("No params or peaks specified!")
        return None
    bins = list(range(int(max(params_data["E_SHORT"]))))
    bin_vals, _bin_edges = np.histogram(params_data["E_SHORT"], bins=bins)
    total_counts1 = sum(bin_vals[index1:])
    total_counts2 = sum(bin_vals[index2:])
    prob = total_counts2 / total_counts1
    sipm.cross_talk.append(prob)
    return prob


def delay_times(params_data, waves_data, min_height, min_dist, width=0):
    all_times = []

    for i, wave in waves_data.iterrows():
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        if len(peaks) > 0:
            times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
            all_times = np.append(all_times, [times])
    M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
    all_dts = M_diag @ all_times
    all_dts = np.delete(all_dts, -1)
    return all_dts


def heights(params_data, wave_data, min_height, min_dist, width=0):
    all_heights = []

    for i, wave in wave_data.iterrows():
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        peak_heights = wave.values[peaks]
        all_heights = np.append(all_heights, [peak_heights])
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
