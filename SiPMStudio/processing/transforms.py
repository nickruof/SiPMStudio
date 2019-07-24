import numpy as np
import pandas as pd
import pywt

from scipy.signal import savgol_filter
from scipy.signal import filtfilt
from statsmodels.robust import mad

from SiPMStudio.processing.functions import butter_bandpass
from SiPMStudio.io.file_settings import read_file

# TODO: come up with way to store waveform timing information and check DataFrame initialisation speed


def adc_to_volts(waves_data, digitizer):
    v_pp = digitizer.v_range
    n_bits = digitizer.adc_bitcount
    processed_waveforms = (v_pp/2**n_bits) * waves_data.to_numpy()
    return pd.DataFrame(data=processed_waveforms, index=waves_data.index, columns=waves_data.columns)


def baseline_subtract(waves_data):
    baseline = np.mean(waves_data.to_numpy(), axis=1)
    baselines_vector = baseline.reshape((baseline.shape[0], 1))
    processed_waveforms = waves_data.to_numpy() - baselines_vector
    return pd.DataFrame(data=processed_waveforms, index=waves_data.index, columns=waves_data.columns)


def savgol(waves_data, window=15, order=2):
    filtered_data = savgol_filter(waves_data.to_numpy(), window, order, axis=1)
    return pd.DataFrame(data=filtered_data, index=waves_data.index, columns=waves_data.columns)


def butter_bandpass_filter(waves_data, digitizer, lowcut, highcut, order=5):
    sample_rate = digitizer.sample_rate
    (b, a) = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    filtered_data = filtfilt(b, a, waves_data.to_numpy(), axis=1)
    return pd.DataFrame(data=filtered_data, index=waves_data.index, columns=waves_data.columns)


def wavelet_denoise(waves_data, wavelet="db1", levels=3, mode="soft"):
    data = waves_data.to_numpy()
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=levels, axis=0)
    sigma = mad(coeffs[-levels], axis=0)
    uthresh = sigma * np.sqrt(2 * np.log(data.shape[1]))
    coeffs[1:] = (pywt.threshold(coeffs[i], value=uthresh, mode=mode) for i in range(1, len(coeffs)))
    filtered_data = pywt.waverec(coeffs, wavelet, axis=0)
    return pd.DataFrame(data=filtered_data, index=waves_data.index, columns=waves_data.columns)


def moving_average(waves_data, box_size=20):
    box = np.ones(box_size) / box_size
    smooth_waves = np.apply_along_axis(lambda wave: np.convolve(wave, box, mode="same"), axis=0, arr=waves_data.to_numpy())
    return pd.DataFrame(data=smooth_waves, index=waves_data.index, columns=waves_data.columns)


def normalize_waves(waves_data, peak_locs):  # path, file_name, settings_option="ph_peaks"):
    # peak_locs = np.array(read_file(path, file_name, file_type="waves")[settings_option])
    diffs = peak_locs[1:] - peak_locs[:-1]
    average_diff = np.mean(diffs)
    norm_data = waves_data.to_numpy() - peak_locs[0]
    norm_data = norm_data / average_diff
    norm_data = norm_data + 1
    return pd.DataFrame(data=norm_data, index=waves_data.index, columns=waves_data.columns)



