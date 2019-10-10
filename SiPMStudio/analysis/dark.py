import numpy as np
import matplotlib.pyplot as plt
import os
import operator
import warnings
import pickle as pk
import pandas as pd

from scipy.sparse import diags
from lmfit import Model
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
import scipy.constants as const
from uncertainties import ufloat, unumpy

from SiPMStudio.processing.functions import gaussian
from SiPMStudio.processing.functions import exponential
from SiPMStudio.analysis.noise import average_power
import SiPMStudio.plots.plots_base as plots_base
import SiPMStudio.plots.plotting as sipm_plt
from SiPMStudio.processing.transforms import savgol

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


def list_files(path, prefix="t2", suffix=".h5"):
    actual_path = os.path.abspath(path)
    all_files = [file for file in sorted(os.listdir(actual_path)) if os.path.isfile(os.path.join(actual_path, file))]
    prefix_files = [file for file in all_files if file.startswith(prefix)]
    suffix_files = [file for file in prefix_files if file.endswith(suffix)]
    return suffix_files


def label_uncertainties(unumpy_array, name):
    for i, element in enumerate(unumpy_array):
        unumpy_array[i].tag = name


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
    peak_locations = []
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
            mu_stderr = []
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
                stderrs = np.sqrt(np.diag(covs))
                gauss_mus.append(coeffs[0])
                mu_stderr.append(stderrs[0])
                gauss_amps.append(coeffs[2])

            peak_locations = unumpy.uarray(gauss_mus, mu_stderr)
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

    return peak_locations


def gain(digitizer, sipm, file_name, params_data=None, waves_data=None):
    pc_peaks = []
    with open(file_name, "rb") as pickle_file:
        pc_peaks = pk.load(pickle_file)
    diffs = pc_peaks[1:] - pc_peaks[:-1]
    gain_average = ufloat((np.mean(diffs[:4])).nominal_value, (np.mean(diffs[:4])).std_dev, "statistical")
    sipm.gain.append(gain_average)
    gain_magnitude = gain_average * digitizer.e_cal/const.e
    sipm.gain_magnitude.append(gain_magnitude)
    return gain_average, gain_magnitude


def dark_count_rate(sipm, height=0.75, distance=50, width=10, bounds=None, params_data=None, waves_data=None, low_counts=False, filt=False, save=False, save_path=None):

    # TODO: Replace hard coded sampling rate of CAENDT5730 with something more generic

    rate = []
    all_dts = []
    if low_counts:
        times = params_data["TIMETAG"].to_numpy()
        all_dts = (times[1:] - times[:-1]) * 10**-3
    else:
        analysis_waves = waves_data
        if filt:
            analysis_waves = savgol(waves_data, 9, 3)
        all_dts = delay_times(params_data, waves_data, height, distance, width)
        for i, wave in enumerate(analysis_waves.to_numpy()):
            peaks, _properties = find_peaks(x=wave, height=height, distance=distance, width=width)
            rate.append(len(peaks) / (len(wave) * 2e-9))
        # pulse_rate
        average_pulse_rate = np.mean(rate)
        error_pulse_rate = np.std(rate)
        sipm.pulse_rate.append(ufloat(average_pulse_rate, error_pulse_rate))

    # exponential fit to delay time histogram
    if bounds is None:
        bounds = [200, 1e4]
    all_dts = np.array(all_dts)
    dts_fit = all_dts[(all_dts > bounds[0]) & (all_dts < bounds[1])]

    [n, bin_edges] = np.histogram(dts_fit, bins=350, range=bounds)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
    log_bins = np.log(n[n > 0])
    fit_centers = bin_centers[n > 0]
    slope, intercept, _r_value, _p_value, stderr = linregress(fit_centers, log_bins)
    slope_param = ufloat(slope, stderr)
    dark_rate = -1*slope_param/1.0e-9
    # exp_model = Model(exponential)
    # params = exp_model.make_params(a=0.001, tau=1100)
    # [n, bin_edges] = np.histogram(dts_fit, bins=350, range=bounds, density=True)
    # centers = (bin_edges[1:]+bin_edges[:-1])/2
    # result = exp_model.fit(n, params, x=centers)

    # time_constant = ufloat(result.params["tau"].value, result.params["tau"].stderr)
    dark_rate = ufloat(dark_rate.nominal_value, dark_rate.std_dev, "statistical")
    sipm.dcr_fit.append(dark_rate)

    if save:
        fig, ax = plt.subplots()
        ax.plot(bin_centers[~nan_places], log_bins[~nan_places])
        ax.plot(bin_centers, slope*bin_centers + intercept)
        plot_index = sipm.dcr_fit.index(dark_rate)
        plt.savefig(save_path+"/dark_count_rate_"+sipm.bias[plot_index]+".png", dpi=300)

    return dark_rate


def dark_photon_rate(sipm, height=0.75, distance=50, width=10, params_data=None, waves_data=None, filt=False, display=False):
    photon_triggers = []
    analysis_waves = waves_data
    if filt:
        analysis_waves = savgol(waves_data, 9, 3)
    for wave in analysis_waves.to_numpy():
        peaks, _properties = find_peaks(x=wave, height=height, distance=distance, width=width)
        round_heights = np.around(wave[peaks])
        photon_triggers.append(np.sum(round_heights))
    average_photon_rate = np.mean(photon_triggers) / (waves_data.shape[1]*2.0e-9)
    stderr_photon_rate = np.std(photon_triggers) / (waves_data.shape[1]*2.0e-9)
    sipm.photon_rate.append(ufloat(average_photon_rate, stderr_photon_rate, "statistical"))
    return average_photon_rate


def cross_talk(sipm, label, params_data=None, waves_data=None):
    energy_data = params_data[label].to_numpy()

    def accumulate_events(data, position):
        counts = np.ones(len(data[data > position]))
        counts_upper = np.ones(len(data[data > (position + 1)]))
        return np.sum(counts), np.sum(counts_upper)
    upper_bounds = accumulate_events(energy_data, 0.5 + sipm.gain_magnitude[-1].s/sipm.gain_magnitude[-1].n)
    middle = accumulate_events(energy_data, 0.5)
    lower_bounds = accumulate_events(energy_data, 0.5 - sipm.gain_magnitude[-1].s/sipm.gain_magnitude[-1].n)

    prob_upper = upper_bounds[1]/upper_bounds[0]
    prob_middle = middle[1]/middle[0]
    prob_lower = lower_bounds[1]/lower_bounds[0]
    sipm.cross_talk.append(ufloat(prob_middle, abs(prob_upper-prob_lower)/2, "statistical"))
    return ufloat(prob_middle, abs(prob_upper-prob_lower)/2)


def delay_times(params_data, waves_data, min_height=0.5, min_dist=50, width=10):
    all_times = []
    for i, wave in enumerate(waves_data.to_numpy()):
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        if len(peaks) > 0:
            peak_list = list(peaks)
            times = map(lambda x: x * 2, peak_list)
            times = map(lambda x: x + params_data.iloc[i, 0]*1e-3, times)
            all_times += times
    M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
    all_dts = M_diag @ all_times
    all_dts = np.delete(all_dts, -1)
    return all_dts


def delay_time_intervals(waves_data, min_height=0.5, min_dist=50, width=10, params_data=None):
    all_dts = []
    for i, wave in enumerate(waves_data.to_numpy()):
        peaks, _properties = find_peaks(x=wave, height=min_height, distance=min_dist, width=width)
        diffs = list((peaks[1:] - peaks[:-1])*2)
        all_dts += diffs
    return np.array(all_dts)


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
