import numpy as np

from scipy import fftpack
from scipy.integrate import trapz


def noise_spectrum(waves_data, digitizer):
    waves_data_fft = fftpack.fft(waves_data, axis=0)
    sample_period = 1 / digitizer.sample_rate
    num_samples = waves_data.shape[1]
    frequencies = np.linspace(0.0, 1.0/(2.0*sample_period), num_samples/2)
    power_spec = 2.0/num_samples * np.abs(waves_data_fft[:num_samples//2])
    return frequencies, power_spec


def average_power(waveform):
    time = np.linspace(0, 2*len(waveform), len(waveform))
    waveform_squared = np.multiply(waveform, waveform)
    average = (1/len(time)) * trapz(waveform_squared, time)
    return average


def snr(waveform, noise):
    noise_average = average_power(noise)
    signal_average = average_power(waveform)
    signal_to_noise = 10 * np.log10((signal_average-noise_average) / noise_average)
    return signal_to_noise

