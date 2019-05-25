import numpy as np

from scipy import fftpack


def noise_spectrum(waves_data, digitizer):
    waves_data_fft = fftpack.fft(waves_data, axis=0)
    sample_period = 1 / digitizer.sample_rate
    num_samples = waves_data.shape[1]
    frequencies = np.linspace(0.0, 1.0/(2.0*sample_period), num_samples/2)
    power_spec = 2.0/num_samples * np.abs(wave_data_fft[:num_samples//2])
    return frequencies, power_spec


