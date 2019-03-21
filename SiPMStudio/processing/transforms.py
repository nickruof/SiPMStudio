import numpy as np
import pandas as pd
import pywt

from scipy.signal import savgol_filter
from scipy.signal import filtfilt
from statsmodels.robust import mad

from SiPMStudio.core import digitizers
from SiPMStudio.core import data_loading
from SiPMStudio.processing.functions import butter_bandpass

def adc_to_volts(waveforms, digitizer):
    V_pp = digitizer.v_range
    n_bits = digitizer.adc_bitcount
    processed_waveforms = np.multiply((V_pp/2**n_bits), waveforms)
    return processed_waveforms

def baseline_subtract(waveforms):
    processed_waveforms = waveforms.sub(waveforms.mean(axis=1), axis=0)
    return processed_waveforms

def savgol(waveforms, window=15, order=2):
    filtered_data = savgol_filter(waveforms.values, window, order, axis=1)
    processed_waveforms = pd.DataFrame(data=filtered_data, index=waveforms.index, columns=waveforms.columns)
    return processed_waveforms

def butter_bandpass_filter(waveforms, digitizer, lowcut, highcut, order=5):
    sample_rate = digitizer.sample_rate
    (b, a) = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    filtered_data = filtfilt(b, a, waveforms.values, axis=1)
    processed_waveforms = pd.DataFrame(data=filtered_data, index=waveforms.index, columns=waveforms.columns)
    return processed_waveforms

def wavelet_denoise(waveforms, wavelet="db1", levels=3, mode="soft"):
    data = waveforms.values
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=levels, axis=0)
    sigma = mad(coeffs[-levels], axis=0)
    uthresh = sigma * np.sqrt(2 * np.log(data.shape[1]))
    coeffs[1:] = (pywt.threshold(coeffs[i], value=uthresh, mode=mode)  for i in range(1, len(coeffs)))
    filtered_data = pywt.waverec(coeffs, wavelet, axis=0)
    processed_waveforms = pd.DataFrame(data=filtered_data, index=waveforms.index, columns=waveforms.columns)
    return processed_waveforms

