import numpy as np
import pywt
import peakutils

from scipy.signal import savgol_filter, filtfilt, find_peaks
from scipy.optimize import curve_fit
from statsmodels.robust import mad
from functools import partial

from SiPMStudio.processing.functions import butter_bandpass, exp_charge, double_exp_release


def adc_to_volts(waves_data, digitizer):
    v_pp = digitizer.v_range
    n_bits = digitizer.adc_bitcount
    processed_waveforms = (v_pp/2**n_bits) * waves_data
    return processed_waveforms


def baseline_subtract(outputs, wf_in, wf_out, degree=1):
    waves_data = outputs[wf_in]
    axis_function = partial(peakutils.baseline, deg=degree)
    baselines = np.apply_along_axis(axis_function, 1, waves_data)
    processed_waveforms = waves_data - baselines
    outputs[wf_out] = processed_waveforms


def savgol(waves_data, window=15, order=2):
    filtered_data = savgol_filter(waves_data, window, order, axis=1)
    return filtered_data


def butter_bandpass_filter(waves_data, digitizer, lowcut, highcut, order=5):
    sample_rate = digitizer.sample_rate
    (b, a) = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    filtered_data = filtfilt(b, a, waves_data, axis=1)
    return filtered_data


def wavelet_denoise(waves_data, wavelet="db1", levels=3, mode="hard"):

    def denoise_function(wave, wavelet_type, num_levels, thresh_mode):
        coeff = pywt.wavedec(data=wave, wavelet=wavelet, level=levels)
        sigma = mad(coeff[-levels])
        uthresh = sigma * np.sqrt(2 * np.log(len(wave)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode=mode) for i in coeff[1:])
        denoised_wave = pywt.waverec(coeff, wavelet)
        return denoised_wave

    axis_function = partial(denoise_function, wavelet_type=wavelet, num_levels=levels, thresh_mode=mode)
    denoised_wave_values = np.apply_along_axis(axis_function, 1, waves_data)
    return denoised_wave_values


def fit_waveforms(outputs, wf_in, wf_out, short_tau, long_tau, charge_up, lback=10, lfor=50, samp=2, max_amp=100):
    output_waveforms = []
    waves_data = outputs[wf_in]
    times = np.arange(0, 2*waves_data.shape[1], 2)
    for i, wave in enumerate(waves_data):
        peak_info = find_peaks(wave, height=250, distance=50, width=4)
        if len(peak_info[0]) == 0:
            output_waveforms.append(np.array([np.mean(wave)]*len(times)))
            continue
        base_wave = wave
        pulses = []
        for j, peak in enumerate(peak_info[0]):
            idx = peak - charge_up[0]*2
            if idx < 0:
                idx = 0
            charge_time, release_time = times[idx:peak], times[peak:peak+lfor]
            charge_form, release_form = base_wave[idx:peak], base_wave[peak:peak+lfor]
            release_coeffs, release_cov = curve_fit(double_exp_release, release_time, release_form,
                                                    p0=[samp*peak, 200, 5e6, short_tau[0], long_tau[0]],
                                                    bounds=([samp*peak-lback, 0, 0, short_tau[1], long_tau[1]], [samp*peak+lback, np.inf, np.inf, short_tau[2], long_tau[2]]))
            charge_coeffs, charge_cov = curve_fit(exp_charge, charge_time, charge_form, p0=[600, samp*peak, charge_up[0]],
                                                 bounds=([0, samp*peak-lback, charge_up[1]], [np.inf, samp*peak+10, charge_up[2]]))
            charge_up_part = exp_charge(times[:peak], *charge_coeffs)
            release_part = double_exp_release(times[peak:], *release_coeffs)
            fit_waveform = np.concatenate((charge_up_part, release_part))
            if max(fit_waveform) < max_amp:
                continue
            else:
                pulses.append(fit_waveform)
                base_wave = base_wave - fit_waveform
        output_waveforms.append(np.sum(pulses, axis=0))
        outputs[wf_out] = np.array(output_waveforms)
