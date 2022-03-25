import re
import numpy as np
import pywt
import peakutils
import warnings
from scipy.optimize.optimize import OptimizeWarning

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

from scipy.signal import savgol_filter, filtfilt, find_peaks
from scipy.optimize import curve_fit
from statsmodels.robust import mad
from functools import partial

from SiPMStudio.processing.functions import butter_bandpass, exp_charge, exp_release, double_exp_release, gaussian


def baseline_subtract(outputs, wf_in, wf_out, degree=1, flip=False):
    waves_data = outputs[wf_in]
    if flip:
        waves_data = -outputs[wf_in]
    else:
        waves_data = outputs[wf_in]
    axis_function = partial(peakutils.baseline, deg=degree)
    baselines = np.apply_along_axis(axis_function, 1, waves_data)
    outputs[wf_out] = waves_data - baselines


def baseline_subtract_simple(outputs, wf_in, wf_out, t_range=[0, 100], other_t_range=[4800, 4900], flip=False):
    waves_data = outputs[wf_in]
    if flip:
        waves_data = -outputs[wf_in]
    def axis_function(waveform):
        average = np.mean(waveform[t_range[0]: t_range[1]])
        baseline = np.repeat(average, repeats=len(waveform))
        return baseline
    baselines = np.apply_along_axis(axis_function, 1, waves_data)
    outputs[wf_out] = waves_data - baselines


def baseline_subtract_gauss(outputs, wf_in, wf_out, sample_range=None, flip=False):
    waves_data = outputs[wf_in]
    if flip:
        waves_data = -outputs[wf_in]
    def axis_function(waveform):
        fit_waveform = waveform
        if sample_range is not None:
            fit_waveform = waveform[sample_range[0]: sample_range[1]]
        n, bins = np.histogram(fit_waveform, bins=100)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        max_loc = np.where(n == max(n))[0][0]
        coeffs, covs = curve_fit(gaussian, bin_centers, n, p0=[bin_centers[max_loc], 100, 100])
        base_value = coeffs[0]
        return np.repeat(base_value, repeats=len(waveform))
    try:
        baselines = np.apply_along_axis(axis_function, 1, waves_data)
        outputs[wf_out] = waves_data - baselines
    except:
        if sample_range is None:
            baseline_subtract_simple(outputs, wf_in, wf_out)
        else:
            baseline_subtract_simple(outputs, wf_in, wf_out, t_range=sample_range, flip=flip)


def savgol(waves_data, window=15, order=2):
    filtered_data = savgol_filter(waves_data, window, order, axis=1)
    return filtered_data


def butter_bandpass_filter(waves_data, digitizer, lowcut, highcut, order=5):
    sample_rate = digitizer.sample_rate
    (b, a) = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    filtered_data = filtfilt(b, a, waves_data, axis=1)
    return filtered_data


def denoise_function(wave, wavelet_type, num_levels, thresh_mode, zero=False):
    coeff = pywt.wavedec(data=wave, wavelet=wavelet_type, level=num_levels)
    sigma = mad(coeff[-num_levels])
    uthresh = 0
    if zero:
        uthresh = 1e6
    else:
        uthresh = sigma * np.sqrt(2 * np.log(len(wave)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode=thresh_mode) for i in coeff[1:])
    denoised_wave = pywt.waverec(coeff, wavelet_type)
    return denoised_wave


def wavelet_denoise(outputs, wf_in, wf_out, wavelet="db1", levels=3, mode="hard", zero=False):
    waves_data = outputs[wf_in]
    axis_function = partial(denoise_function, wavelet_type=wavelet, num_levels=levels, thresh_mode=mode, zero=zero)
    denoised_wave_values = np.apply_along_axis(axis_function, 1, waves_data)
    outputs[wf_out] = denoised_wave_values


def fit_waveforms(outputs, wf_in, wf_out, short_tau, long_tau, charge_up, lback=10, lfor=50, samp=2, max_amp=100):
    output_waveforms = []
    waves_data = outputs[wf_in]
    times = np.arange(0, 2*waves_data.shape[1], 2)
    for i, wave in enumerate(waves_data):
        new_wave = denoise_function(wave, "db1", 3, "hard")
        peak_info = find_peaks(wave, height=250, distance=10, width=4)
        if len(peak_info[0]) == 0:
            output_waveforms.append(np.array([0]*len(times)))
            if i == waves_data.shape[0] - 1:
                outputs[wf_out] = np.array(output_waveforms)
            continue
        base_wave = wave
        pulses = []
        for j, peak in enumerate(peak_info[0]):
            idx = peak - charge_up[0]*2
            if idx < 0:
                idx = 0

            release_coeffs = None
            charge_coeffs = None
            try:
                charge_time, release_time = times[idx:peak], times[peak:peak+lfor]
                charge_form, release_form = base_wave[idx:peak], base_wave[peak:peak+lfor]
                release_coeffs, release_cov = curve_fit(double_exp_release, release_time, release_form,
                                                        p0=[samp*peak, 200, 5e6, short_tau[0], long_tau[0]],
                                                        bounds=([samp*peak-lback, 0, 0, short_tau[1], long_tau[1]],
                                                        [samp*peak+lback, np.inf, np.inf, short_tau[2], long_tau[2]]))
                charge_coeffs, charge_cov = curve_fit(exp_charge, charge_time, charge_form, p0=[600, samp*peak, charge_up[0]],
                                                    bounds=([0, samp*peak-lback, charge_up[1]], [np.inf, samp*peak+10, charge_up[2]]))
                charge_up_part = exp_charge(times[:peak], *charge_coeffs)
                release_part = double_exp_release(times[peak:], *release_coeffs)
                fit_waveform = np.concatenate((charge_up_part, release_part))
                if (release_coeffs[1]+release_coeffs[2]) < max_amp:
                    continue
                else:
                    pulses.append(fit_waveform)
                    base_wave = base_wave - fit_waveform
            except RuntimeError:
                continue
        if len(pulses) > 0:
            output_waveforms.append(np.sum(pulses, axis=0))
        else:
            output_waveforms.append(np.array([0]*len(times)))
        outputs[wf_out] = np.array(output_waveforms)
