import numpy as np
import pandas as pd
import pywt

from scipy.signal import savgol_filter
from scipy.signal import filtfilt
from statsmodels.robust import mad

from SiPMStudio.processing.functions import butter_bandpass
from SiPMStudio.analysis.dark import heights


def adc_to_volts(waves_data, digitizer):
    v_pp = digitizer.v_range
    n_bits = digitizer.adc_bitcount
    processed_waveforms = np.multiply((v_pp/2**n_bits), waves_data)
    return processed_waveforms


def baseline_subtract(waves_data):
    processed_waveforms = waves_data.sub(waves_data.mean(axis=1), axis=0)
    return processed_waveforms


def savgol(waves_data, window=15, order=2):
    filtered_data = savgol_filter(waveforms.values, window, order, axis=1)
    processed_waveforms = pd.DataFrame(data=filtered_data, index=waveforms.index, columns=waveforms.columns)
    return processed_waveforms


def butter_bandpass_filter(waves_data, digitizer, lowcut, highcut, order=5):
    sample_rate = digitizer.sample_rate
    (b, a) = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    filtered_data = filtfilt(b, a, waveforms.values, axis=1)
    processed_waveforms = pd.DataFrame(data=filtered_data, index=waveforms.index, columns=waveforms.columns)
    return processed_waveforms


def wavelet_denoise(waves_data, wavelet="db1", levels=3, mode="soft"):
    data = waveforms.values
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=levels, axis=0)
    sigma = mad(coeffs[-levels], axis=0)
    uthresh = sigma * np.sqrt(2 * np.log(data.shape[1]))
    coeffs[1:] = (pywt.threshold(coeffs[i], value=uthresh, mode=mode)  for i in range(1, len(coeffs)))
    filtered_data = pywt.waverec(coeffs, wavelet, axis=0)
    processed_waveforms = pd.DataFrame(data=filtered_data, index=waveforms.index, columns=waveforms.columns)
    return processed_waveforms


def moving_average(waves_data, box_size=20):
    smooth_waves = []
    box = np.ones(box_size) / box_size
    for wave in waves_data.values:
        smooth_wave = np.convolve(wave, box, mode="same")
        smooth_waves.append(smooth_wave)
    processed_waveforms = pd.DataFrame(data=smooth_waves, index=waves_data.index, columns=waves_data.columns)
    return processed_waveforms


def normalize_waves(waves_data, path, file_name):
    peak_locs = read_file(path, file_name)["peak_waves"]
    diffs = peak_locs[1:] - peak_locs[:-1]
    average_diff = np.mean(diffs)
    normalized = np.subtract(waves_data, peak_locs[0])
    normalized = np.divide(normalized, average_diff)
    normalized = np.add(normalized, 1)
    return normalized



