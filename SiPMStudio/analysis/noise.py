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


def average_power(waves_data):
    time = np.linspace(0, 2*waves_data.shape[1], waves_data.shape[1])
    waves_squared = np.multiply(waves_data, waves_data)
    delta_t = time[-1] - time[0]
    average = (1/delta_t) * trapz(waves_squared, time, axis=1)
    return average


def snr(waves_data, noise=None, noise_power=None):
    noise_average = 0
    if noise is None and noise_power is not None:
        noise_average = noise_power * np.ones(waves_data.shape[0])
    elif (noise is not None) and (noise_power is None):
        noise_average = average_power(noise)
    else:
        raise AttributeError("Specify noise waveform or a noise power value!")
    signal_average = average_power(waves_data)
    inner_term = np.divide(signal_average-noise_average, noise_average)
    signal_to_noise = 10 * np.log10(inner_term)
    return signal_to_noise
