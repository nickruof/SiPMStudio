import numpy as np
import pandas as pd
import os

from scipy.sparse import diags
from scipy.stats import expon

from SiPMStudio.core import data_loading
from SiPMStudio.core import digitizers
from SiPMStudio.core import devices
from SiPMStudio.calculations.helpers import detect_peaks
from SiPMStudio import functions


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


def time_interval(params_data):
    interval = params_data["TIMETAG"].iloc[-1] - params_data["TIMETAG"].iloc[0]
    interval = interval * 1.0e-12
    return interval


def fit_multi_gauss(params_data, min_dist, min_height, display=False):
    bins = np.linspace(start=0, stop=max(calcs),
                       num=int(max(calcs))
    bin_vals, _bin_edges = np.histogram(calcs, bins=bins, density=True)
    peaks = detect_peaks(bin_vals, mpd=min_dist, mph=min_height)
    amplitudes = bin_vals[peaks]
    sigmas = [17]*len(peaks) #method needed to avoid hard coded sigma guess
    guess = []
    for i, peak in enumerate(peaks):
        guess.append(peak)
        guess.append(amplitudes[i])
        guess.append(sigmas[i])
    popt, pcov = curve_fit(multi_gauss, xdata=bins[:-1], ydata=bin_vals, p0=guess)
    fit = multi_gauss(bins[:-1], *popt)
    if display:
        plt.figure()
        plt.plot(bins[:-1], fit, color="red")
    return popt


def gain(params, digitizer, sipm, sum_len=1):
    diffs = []
    gain_average = 1
    for i in range(0, len(params)-3, 3):
        diffs.append(params[i+3] - params[i])
    gain_average = sum(diffs[0:sum_len]) / float(len(diffs[0:sum_len]))
    sipm.gain.append(gain_average)
    gain_magnitude = gain_average * digitizer.e_cal/1.6e-19
    sipm.gain_magnitude.append(gain_magnitude)    
    return gain_average, gain_magnitude


def pulse_rate(wave_data, sipm, min_height, min_dist):
    pulse_rate = []
    for i, wave in wave_data.iterrows():
        peaks = detect_peaks(wave, mph=min_height, mpd=min_dist)
        pulse_rate.append(len(peaks) / (len(wave)*2e-9))
    average_pulse_rate = sum(pulse_rate) / len(pulse_rate)
    sipm.pulse_rate.append(average_pulse_rate)
    return average_pulse_rate


def dcr_exp_fit(dts, sipm, bounds):
    if bounds is None:
        bounds=[0, 1e5]
    dts_fit = dts[(dts > bounds[0]) & (dts < bounds[1])]
    exp_fit = expon.fit(dts_fit)
    sipm.dcr_fit.append(1/(exp_fit[1]*1e-9))
    return 1/(exp_fit[1]*1e-9)


def excess_charge_factor(sipm):
    return np.divide(sipm.pulse_rate, sipm.dcr_fit)


def cross_talk(params_data, params, sipm):
    index1 = int(params[0] - sipm.gain[-1]/2)
    index2 = int(params[3] - sipm.gain[-1]/2)
    bins = list(range(int(max(params_data["E_SHORT"]))))
    bin_vals, _bin_edges = np.histogram(params_data["E_SHORT"], bins=bins)
    total_counts1 = sum(bin_vals[index1:])
    total_counts2 = sum(bin_vals[index2:])
    prob = total_counts2 / total_counts1
    sipm.cross_talk.append(prob)
    return prob


def delay_times(params_data, wave_data, min_height, min_dist):
    all_times = []

    for i, wave in wave_data.iterrows():
        peaks = detect_peaks(wave, mph=min_height, mpd=min_dist)
        times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
        all_times = np.append(all_times, [times])
    M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
    all_dts = M_diag @ all_times
    all_dts = np.delete(all_dts, -1)
    return all_dts


def heights(wave_data, min_height, min_dist):
    all_heights = []

    for i, wave in wave_data.iterrows():
        peaks = detect_peaks(wave, mph=min_height, mpd=min_dist)
        peak_heights = wave_data[i].values[peaks]
        all_heights = np.append(all_heights, [peak_heights])
    all_heights = np.delete(all_heights, -1)
    return all_heights


def delay_time_vs_height(params_data, wave_data, min_height, min_dist):
    all_dts = []
    all_heights = []
    all_times = []

    for i, wave in wave_data.iterrows():
        peaks = detect_peaks(wave, mph=min_height, mpd=min_dist)
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
