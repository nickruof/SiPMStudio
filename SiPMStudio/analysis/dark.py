import numpy as np
import matplotlib.pyplot as plt
import os
import operator
import warnings
import pickle as pk

from scipy.sparse import diags
from scipy.stats import expon
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from SiPMStudio.processing.functions import gaussian
from SiPMStudio.processing.functions import multi_gauss
from SiPMStudio.analysis.noise import average_power
import SiPMStudio.plots.plots_base as plots_base
import SiPMStudio.plots.plotting as sipm_plt

warnings.filterwarnings("ignore", "PeakPropertyWarning: some peaks have a width of 0")


def collect_files(path, digitizer, data_dir="UNFILTERED"):
    dirs_array = []
    file_array = []
    for dirpath, dirnames, filenames in os.walk(path):
        if dirnames:
            dirs_array.append(dirnames)

    runs = []
    waves = []
    noise = []
    if len(dirs_array) == 0:
        raise LookupError("No Directories Found!")
    elif len(dirs_array[1]) == 0:
        raise FileNotFoundError("No Files Found in the Directory!")
    for name in dirs_array[1]:
        if "runs_" in name:
            runs.append(name)
        elif "waves_" in name:
            waves.append(name)
        elif "noise_" in name:
            noise.append(name)

    run_files = []
    wave_files = []
    noise_files = []

    for run in runs:
        data_path = path+"/DAQ/"+run+"/"+data_dir
        files = os.listdir(data_path)
        file_targets = []
        os.chdir(data_path)
        for file in files:
            if digitizer.file_header in file:
                run_files.append(data_path+"/"+file)
    for wave in waves:
        data_path = path+"/DAQ/"+wave+"/"+data_dir
        files = os.listdir(data_path)
        file_targets = []
        os.chdir(data_path)
        for file in files:
            if digitizer.file_header in file:
                wave_files.append(data_path+"/"+file)
    for nose in noise:
        data_path = path+"/DAQ/"+nose+"/"+data_dir
        files = os.listdir(data_path)
        file_targets = []
        os.chdir(data_path)
        for file in files:
            if digitizer.file_header in file:
                noise_files.append(data_path+"/"+file)

    return run_files, wave_files, noise_files


def list_files(path):
    all_files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    h5_files = [file for file in all_files if file.endswith(".h5")]
    return h5_files


def time_interval(params_data, waves_data=None):
    interval = params_data["TIMETAG"].iloc[-1] - params_data["TIMETAG"].iloc[0]
    interval = interval * 1.0e-12
    return interval


def noise_power(waves_data, sipm, params_data=None):
    powers = average_power(waves_data)
    total_average = np.mean(powers)
    sipm.noise_power.append(total_average)
    return total_average


def signal_power(params_data, waves_data, sipm, energy_label="ENERGY"):
    energy_data = params_data[energy_label]
    one_peak = (energy_data > 0.9) & (energy_data < 1.1)
    two_peak = (energy_data > 1.9) & (energy_data < 2.1)
    three_peak = (energy_data > 2.9) & (energy_data < 3.1)
    four_peak = (energy_data > 3.9) & (energy_data < 4.1)
    waves_1 = waves_data.to_numpy()[one_peak]
    waves_2 = waves_data.to_numpy()[two_peak]
    waves_3 = waves_data.to_numpy()[three_peak]
    waves_4 = waves_data.to_numpy()[four_peak]

    averages = []
    for waves in [waves_1, waves_2, waves_3, waves_4]:
        signal_sections = waves.T[:200]
        signal_sections = signal_sections.T
        powers = average_power(signal_sections)
        averages.append(np.mean(powers))
    sipm.signal_power.append(averages)
    return averages


def spectrum_peaks(params_data, waves_data=None, n_bins=500, hist_range=None, min_dist=0.0, min_height=0.0, width=0.0,
                   display=False, fit_peaks=False):

    peaks = []
    bin_edges = []
    if display:
        fig, ax = plt.subplots()
        [bin_vals, bin_edges, _patches] = plots_base.plot_hist(ax, [params_data], bins=n_bins, x_range=hist_range, density=False)
        bin_width = bin_edges[0][1] - bin_edges[0][0]
        peaks, _properties = find_peaks(bin_vals[0], height=min_height, distance=min_dist, width=width)
        print(str(len(peaks)) + " peaks found!")
        bin_centers = (bin_edges[0][:-1]+bin_edges[0][1:])/2
        if fit_peaks:
            gauss_mus = []
            gauss_amps = []
            for peak in peaks:
                start = peak - int(min_dist/2)
                stop = peak + int(min_dist/2)
                if start < 0:
                    start = 0
                if stop > len(bin_centers) - 1:
                    stop = len(bin_centers) - 1
                x_region = bin_centers[start:stop]
                y_region = bin_vals[0][start:stop]
                coeffs, covs = curve_fit(gaussian, x_region, y_region, [bin_centers[peak], 10, bin_vals[0][peak]])
                gauss_mus.append(coeffs[0])
                gauss_amps.append(coeffs[2])

            ax.plot(gauss_mus, gauss_amps, "+r")
            ax.set_yscale("log")
            fig.show()
        else:
            ax.plot(bin_centers[peaks], bin_vals[0][peaks], "r+")
            ax.set_yscale("log")
            fig.show()
    else:
        [bin_vals, bin_edges, _patches] = plots_base.plot_hist(ax, [params_data], bins=n_bins, x_range=hist_range, density=False)
        bin_width = bin_edges[0][1] - bin_edges[0][0]
        peaks, _properties = find_peaks(bin_vals[0], height=min_height, distance=min_dist, width=width)

    x_values = np.array(bin_edges[0][:-1])
    return x_values[peaks]


def gain(digitizer, sipm, file_name, params_data=None, waves_data=None):
    pc_peaks = []
    with open(file_name, "rb") as pickle_file:
        pc_peaks = pk.load(pickle_file)
    diffs = pc_peaks[1:] - pc_peaks[:-1]
    gain_average = np.mean(diffs[:4])
    sipm.gain.append(gain_average)
    gain_magnitude = gain_average * digitizer.e_cal/1.6e-19
    sipm.gain_magnitude.append(gain_magnitude)    
    return gain_average, gain_magnitude


def dark_count_rate(sipm, bounds=None, params_data=None, waves_data=None, low_counts=False, display=False):

    # TODO: Replace hard coded sampling rate of CAENDT5730 with something more generic

    rate = []
    all_dts = []
    if low_counts:
        all_dts = delay_times(params_data, waves_data, 0.5, 50, 10)
        for i, wave in enumerate(waves_data.to_numpy()):
            peaks, _properties = find_peaks(x=wave, height=0.75, distance=50, width=10)
            rate.append(len(peaks) / (len(wave) * 2e-9))
    else:
        for i, wave in enumerate(waves_data.to_numpy()):
            peaks, _properties = find_peaks(x=wave, height=0.75, distance=50, width=10)
            rate.append(len(peaks) / (len(wave) * 2e-9))
            if len(peaks) >= 2:
                times = map(operator.sub, peaks[1:], peaks[:-1])
                times = map(lambda x: x * 2, times)
                all_dts += times

    # pulse_rate
    average_pulse_rate = np.mean(rate)
    sipm.pulse_rate.append(average_pulse_rate)

    # exponential fit to delay time histogram
    if bounds is None:
        bounds = [0, 1e5]
    all_dts = np.array(all_dts)
    dts_fit = all_dts[(all_dts > bounds[0]) & (all_dts < bounds[1])]
    exp_fit = expon.fit(dts_fit)
    sipm.dcr_fit.append(1/(exp_fit[1]*1e-9))

    if display:
        fig, ax = plt.subplots()
        sipm_plt.delay_times(ax, dts=all_dts, fit=True)
        plt.show()
        plt.close()

    return average_pulse_rate, 1/(exp_fit[1]*1e-9)


def excess_charge_factor(sipm, params_data=None, waves_data=None):
    return np.divide(sipm.pulse_rate, sipm.dcr_fit)


def cross_talk(sipm, label, params_data=None, waves_data=None):
    energy_data = params_data[label].to_numpy()
    counts_05 = np.ones(len(energy_data[energy_data > 0.5]))
    counts_15 = np.ones(len(energy_data[energy_data > 1.5]))
    total_counts1 = np.sum(counts_05)
    total_counts2 = np.sum(counts_15)
    prob = total_counts2 / total_counts1
    sipm.cross_talk.append(prob*100)
    return prob


def delay_times(params_data, waves_data, min_height=0.5, min_dist=50, width=10):
    all_times = []
    for i, wave in enumerate(waves_data.to_numpy()):
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        if len(peaks) > 0:
            times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
            all_times = np.append(all_times, [times])
    M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
    all_dts = M_diag @ all_times
    all_dts = np.delete(all_dts, -1)
    return all_dts


def trigger_delay_times(waves_data, params_data=None, min_height=0.5, min_dist=50, width=4):
    all_times = []
    all_heights = []
    dict_cut = False
    for i, wave in enumerate(waves_data.to_numpy()):
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        if len(peaks) >= 2:
            all_times.append(2*(peaks[1] - peaks[0]))
            all_heights.append(wave[peaks[1]])
        else:
            all_times.append(0)
            all_heights.append(0)
    return np.array(all_times), np.array(all_heights)


def triggered_heights(waves_data, triggered_index=24):
    return waves_data.iloc[:, triggered_index]


def heights(waves_data, min_height, min_dist, width=0):
    all_heights = []

    for wave in waves_data.to_numpy():
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        if len(peaks) > 0:
            for height in wave[peaks]:
                all_heights.append(height)
    all_heights = np.array(all_heights)
    all_heights = np.delete(all_heights, -1)
    return all_heights


def delay_time_vs_height(params_data, waves_data, min_height=0.5, min_dist=50, width=4):
    all_dts = []
    all_heights = []
    all_times = []

    for i, wave in enumerate(waves_data.to_numpy()):
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
        peak_heights = waves_data.iloc[i, :].to_numpy()[peaks]
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
