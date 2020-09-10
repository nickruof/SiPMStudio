import numpy as np
import pandas as pd
import pywt
import peakutils

from scipy.signal import savgol_filter, filtfilt, wiener, deconvolve
from statsmodels.robust import mad
from functools import partial

from SiPMStudio.processing.functions import butter_bandpass, exp_func, double_exp

# TODO: come up with way to store waveform timing information and check DataFrame initialisation speed


def adc_to_volts(waves_data, digitizer):
    v_pp = digitizer.v_range
    n_bits = digitizer.adc_bitcount
    processed_waveforms = (v_pp/2**n_bits) * waves_data.to_numpy()
    return pd.DataFrame(data=processed_waveforms, index=waves_data.index, columns=waves_data.columns)


# def baseline_subtract(waves_data):
#    baseline = np.mean(waves_data.to_numpy()[:, 0:45], axis=1)
#    baselines_vector = baseline.reshape((baseline.shape[0], 1))
#    processed_waveforms = waves_data.to_numpy() - baselines_vector
#    return pd.DataFrame(data=processed_waveforms, index=waves_data.index, columns=waves_data.columns)


def baseline_subtract(waves_data, degree=1):
    axis_function = partial(peakutils.baseline, deg=degree)
    baselines = np.apply_along_axis(axis_function, 1, waves_data.to_numpy())
    processed_waveforms = waves_data.to_numpy() - baselines
    return pd.DataFrame(data=processed_waveforms, index=waves_data.index, columns=waves_data.columns)


def savgol(waves_data, window=15, order=2):
    filtered_data = savgol_filter(waves_data.to_numpy(), window, order, axis=1)
    return pd.DataFrame(data=filtered_data, index=waves_data.index, columns=waves_data.columns)


def butter_bandpass_filter(waves_data, digitizer, lowcut, highcut, order=5):
    sample_rate = digitizer.sample_rate
    (b, a) = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    filtered_data = filtfilt(b, a, waves_data.to_numpy(), axis=1)
    return pd.DataFrame(data=filtered_data, index=waves_data.index, columns=waves_data.columns)


def wavelet_denoise(waves_data, wavelet="db1", levels=3, mode="hard"):

    def denoise_function(wave, wavelet_type, num_levels, thresh_mode):
        coeff = pywt.wavedec(data=wave, wavelet=wavelet, level=levels)
        sigma = mad(coeff[-levels])
        uthresh = sigma * np.sqrt(2 * np.log(len(wave)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode=mode) for i in coeff[1:])
        denoised_wave = pywt.waverec(coeff, wavelet)
        return denoised_wave

    axis_function = partial(denoise_function, wavelet_type=wavelet, num_levels=levels, thresh_mode=mode)
    denoised_wave_values = np.apply_along_axis(axis_function, 1, waves_data.to_numpy())
    new_columns = [str(i) for i in range(denoised_wave_values.shape[1])]
    return pd.DataFrame(data=denoised_wave_values, index=waves_data.index, columns=new_columns)


def moving_average(waves_data, box_size=20):
    box = np.ones(box_size) / box_size
    smooth_waves = np.apply_along_axis(lambda wave: np.convolve(wave, box, mode="same"), axis=0, arr=waves_data.to_numpy())
    return pd.DataFrame(data=smooth_waves, index=waves_data.index, columns=waves_data.columns)


def deconvolve_waves(waves_data, short_tau, long_tau):
    x_samples = np.linspace(0, 2*waves_data.shape[1], waves_data.shape[1])
    transfer = None
    if short_tau == 0.0:
        transfer = exp_func(x_samples[50:500], 1, 0, long_tau, 0)
    elif long_tau == 0.0:
        transfer = exp_func(x_samples[50:500], 1, 0, short_tau, 0)
    else:
        transfer = double_exp(x_samples[50:500], 0.5, 0.1, 0, short_tau, long_tau, 0)

    def deconvolve_waveform(waveform, transfer_func):
        waveform_wiener = wiener(waveform, 20)
        waveform_deconv = deconvolve(waveform_wiener, transfer_func)
        buffer_length = len(waveform) - len(waveform_deconv[0])
        output_wave = np.append(waveform_deconv[0], [0]*buffer_length)
        return output_wave

    deconvolve_function = partial(deconvolve_waveform, transfer_func=transfer)
    deconv_waves = np.apply_along_axis(deconvolve_function, 1, waves_data.to_numpy())
    return pd.DataFrame(data=deconv_waves, index=waves_data.index, columns=waves_data.columns)


def normalize_waves(waves_data, peak_locs):
    if len(peak_locs) == 0:
        return waves_data
    diffs = peak_locs[1:] - peak_locs[:-1]
    average_diff = np.mean(diffs)
    norm_data = waves_data.to_numpy()
    norm_data = norm_data / average_diff
    return pd.DataFrame(data=norm_data, index=waves_data.index, columns=waves_data.columns)



