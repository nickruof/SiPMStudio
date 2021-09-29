import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings
import math

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

from SiPMStudio.processing.functions import gaussian, rise_func
import SiPMStudio.plots.plots_base as plots_base
from SiPMStudio.processing.transforms import savgol

warnings.filterwarnings("ignore", "PeakPropertyWarning: some peaks have a width of 0")


def time_interval(params_data, waves_data=None):
    interval = params_data["TIMETAG"].iloc[-1] - params_data["TIMETAG"].iloc[0]
    interval = interval * 1.0e-12
    return interval


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


def current_waveforms(waveforms, vpp=2, n_bits=14):
    return waveforms * (vpp / 2 ** n_bits) * (1000 / 31.05) * 1.0e-6


def integrate_current(current_forms, lower_bound=0, upper_bound=200, sample_time=2e-9):
    return np.sum(current_forms.T[lower_bound:upper_bound].T, axis=1)*sample_time


def rando_integrate_current(current_forms, width, sample_time=2e-9):
    start_range = width
    stop_range = current_forms.shape[1] - width - 1
    start = np.random.randint(start_range, stop_range)
    stop = start + width
    return np.sum(current_forms.T[start:stop].T, axis=1)*sample_time


def integrate_slices(current_forms, width, window_size, sample_time=2e-9):
    steps = math.floor((current_forms.shape[1] - 2*window_size) / window_size)
    charges = []
    for i in range(steps):
        start = window_size * (i+1)
        stop = start + width
        charges += list(np.sum(current_forms.T[start:stop].T, axis=1)*sample_time)
    return np.array(charges)


def wave_peaks(waveforms, height=500, distance=5):
    all_peaks = []
    all_heights = []
    for waveform in tqdm.tqdm(waveforms, total=len(waveforms)):
        peak_locs = find_peaks(waveform, height=height, distance=distance)[0]
        heights = waveform[peak_locs]
        if len(heights) == len(peak_locs):
            all_peaks.append(peak_locs)
            all_heights.append(heights)
    return np.asarray(all_peaks, dtype=object), np.asarray(all_heights, dtype=object)


def cross_talk_frac(heights, min_height=0.5, max_height=1.50):
    one_pulses = 0
    other_pulses = 0
    for height_set in tqdm.tqdm(heights, total=len(heights)):
        if len(height_set) > 0:
            if (height_set[0] > min_height) & (height_set[0] < max_height):
                one_pulses += 1
            else:
                other_pulses += 1
        else:
            continue
    return other_pulses / (one_pulses + other_pulses)


def cross_talk_frac_v2(peaks, peak_errors, charges):
    charge_diff = 0
    if any(np.isnan(peak_errors)) | any(np.isinf(peak_errors)):
        charge_diff = peaks[1] - peaks[0]
    else:
        diffs = peaks[1:] - peaks[:-1]
        errors_squared = peak_errors**2
        errors = np.sqrt(errors_squared[1:] + errors_squared[:-1])
        weights = 1 / errors
        charge_diff = np.average(diffs, weights=weights)
    one_charges = charges[(charges > (charge_diff/2)) & (charges < (3*charge_diff/2))]
    other_charges = charges[charges > (3*charge_diff/2)]
    return len(one_charges) / (len(one_charges) + len(other_charges))


def afterpulsing_frac(waves, peaks, heights, display=False, fit_range=None):
    if fit_range is None:
        fit_range = [[25, 200], [0.25, 0.75]]
    ap_waves, ap_peaks, ap_heights = waveform_find(waves, peaks, heights, len(waves), fit_range[0], fit_range[1])
    ap_fraction = len(ap_waves)/len(waves)
    times = []
    all_heights = []
    for i, peak in enumerate(ap_peaks):
        if peak[1] < 125:
            times.append(2*(peak[1] - peak[0]))
            all_heights.append(ap_heights[i][1])
    times = np.array(times)
    all_heights = np.array(all_heights)
    fit_mask = (times > fit_range[0][0]) & (times < fit_range[0][1]) & (all_heights > fit_range[1][0]) \
               & (all_heights < fit_range[1][1])
    time_fit = times[fit_mask]
    height_fit = all_heights[fit_mask]
    if (len(time_fit) < 4) & (len(height_fit) < 4):
        return 0, 0
    coeffs_ap, covs_ap = curve_fit(rise_func, time_fit, height_fit, p0=[1, 0, 80, 0])
    t_rec = coeffs_ap[2]
    if display:
        t_plot = np.linspace(10, 200, 500)
        plt.figure()
        plt.scatter(times, all_heights, s=1, label="Data")
        plt.plot(t_plot, rise_func(t_plot, *coeffs_ap), color="magenta", alpha=0.75, label=r"$t_{rec}=$ "+str(round(t_rec))+" ns")
        plt.xlabel("Inter-times (ns)")
        plt.ylabel("Amplitude (P.E.)")
        plt.legend()
    return ap_fraction, t_rec


def excess_charge_factor(norm_charges, min_charge=0.5, max_charge=1.5):
    primary_charge = norm_charges[(norm_charges > min_charge) & (norm_charges < max_charge)]
    ecf = np.mean(norm_charges) / np.mean(primary_charge)
    return ecf
