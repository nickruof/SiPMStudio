import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from SiPMStudio.processing.functions import butter_bandpass
from SiPMStudio.processing.functions import multi_gauss
from SiPMStudio.calculations.helpers import detect_peaks

from scipy import fftpack
from scipy.signal import freqz

sns.set_style("whitegrid")

def plot_butter_response(digitizer, lowcut, highcut, order=5):
    (b, a) = butter_bandpass(lowcut, highcut, digitizer.sample_rate, order=order)
    (w, h) = freqz(b, a, worN=2000)
    plt.figure()
    plt.plot(((fs * 0.5)/np.pi) * w, abs(h), label="order = %d" % order)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    plt.show()

def plot_FFT(digitizer, waveform):
    wave_fft = fftpack.fft(waveform)
    N = len(waveform)
    sample_period = 1 / digitizer.sample_rate
    x_fft = np.linspace(0.0, 1.0/(2.0*sample_period), N/2)
    y_fft = 2.0/N * np.abs(wave_fft[:N//2])
    fft_norm = np.linalg.norm(y_fft)
    y_fft_norm = [element/fft_norm for element in y_fft]
    plt.figure()
    plt.plot(x_fft, y_fft_norm)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (%)")
    plt.show()

def plot_waveform(waveform, find_peaks=False, min_dist, min_height):
    plt.figure()
    plt.plot(waveform.index.values, waveform)
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (V)")

    if find_peaks:
        peaks = detect_peaks(waveform, mph=min_height, mpd=min_dist)
        peak_heights = waveform[peaks]
        peak_times = waveform.index.values[peaks]
        plt.plot(peak_times, peak_heights, "r.")

def plot_waveforms(waveforms):
    ax = self.waveforms.plot()
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (V)")

def pc_spectrum(hist_array, bins, params, log=False):
    sns.set_style("white")
    if not bins:
        bins = [i for i in range(int(max(hist_array)))]
    plt.hist(hist_array, bins=bins, edgecolor="none")
    plt.xlabel("ADC")
    plt.ylabel("Counts")
    if log:
        plt.yscale("log")
    if params:
        




