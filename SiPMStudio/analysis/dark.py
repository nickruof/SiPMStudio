import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings
import math

from scipy.sparse import diags
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
from uncertainties import ufloat, unumpy

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


def amplitudes(heights):
    amps = []
    for i, height in tqdm.tqdm(enumerate(heights), total=len(heights)):
        amps += list(height)
    return np.array(amps)


def waveform_find(waveforms, peak_locations, heights, n_waveforms, dt_range, pe_range):
    output_waveforms = []
    output_peaks = []
    output_heights = []
    for i, wave in enumerate(tqdm.tqdm(waveforms, total=len(waveforms))):
        dt = 0
        height = 0
        if len(peak_locations[i]) < 2:
            continue
        if (heights[i][0] < 0.98) | (heights[i][0] > 1.11):
            continue
        else:
            dt = 2*(peak_locations[i][1] - peak_locations[i][0])
            height = heights[i][1]
            if (dt > dt_range[0]) & (dt < dt_range[1]) & (height > pe_range[0]) & (height < pe_range[1]):
                output_waveforms.append(wave)
                output_peaks.append(peak_locations[i])
                output_heights.append(heights[i])
            if len(output_waveforms) == n_waveforms:
                return output_waveforms, output_peaks, output_heights
    return output_waveforms, output_peaks, output_heights


def dark_count_rate(waves, times, peaks, heights, pe_lim=0.5, sample_size=2, exclude=None, region=None, bins=1000, display=False):
    all_times = []
    all_heights = []
    if region is None:
        region = [1e3, 1e6]
    for i, peak in tqdm.tqdm(enumerate(peaks), total=len(peaks)):
        if len(peak) == 0:
            continue
        time_values = sample_size*peak + times[i] #sample size in ns
        height_values = heights[i]
        all_times += list(time_values[height_values > pe_lim])
        all_heights += list(height_values[height_values > pe_lim])
    dts = np.array(all_times)[1:] - np.array(all_times)[:-1]
    out_heights = np.array(all_heights)[1:]
    n, bin_edges = np.histogram(dts, bins=bins, range=region)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    if exclude:
        n = n[(bin_centers < exclude[0]) | (bin_centers > exclude[1])]
        bin_centers = bin_centers[(bin_centers < exclude[0]) | (bin_centers > exclude[1])]
    slope, intercept, pvalue, rvalue, stderr = linregress(bin_centers[n > 0], np.log(n[n > 0]))
    slope_param = ufloat(slope, stderr)
    dark_rate = 1 / (abs(1 / slope_param) * 1e-9) / 1000
    if display:
        fig, ax = plt.subplots(1, 2)
        ax[0].hist(dts, bins=bins, range=region, edgecolor="none")
        ax[0].set_xlabel("Inter-times (ns)")
        ax[0].set_ylabel("Counts")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")

        ax[1].step(bin_centers[n > 0], np.log(n[n > 0]))
        ax[1].plot(bin_centers, slope*bin_centers+intercept)
        ax[1].set_xlabel("Time (ns)")
        ax[1].set_ylabel("Log Counts")
        ax[1].legend([str(round(dark_rate.n))+" kHz"])
    return dark_rate.n, dark_rate.s


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
