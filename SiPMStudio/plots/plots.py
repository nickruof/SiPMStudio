import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from SiPMStudio.processing.functions import butter_bandpass
from SiPMStudio.processing.functions import multi_gauss
from SiPMStudio.calculations.helpers import detect_peaks

from scipy import fftpack
from scipy.signal import freqz
from scipy.stats import linregress

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

def plot_waveform(waveform, find_peaks, min_dist, min_height):
    time = np.linspace(0, 2*len(waveform)-1, len(waveform))
    plt.figure()
    plt.plot(time, waveform)
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (V)")

    if find_peaks:
        peaks = detect_peaks(waveform, mph=min_height, mpd=min_dist)
        print(peaks)
        peak_heights = waveform.values[peaks]
        peak_times = time[peaks]
        plt.plot(peak_times, peak_heights, "r.")
    plt.show()

def plot_waveforms(waveforms):
    ax = self.waveforms.plot()
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (V)")
    ax.set_xlim([0, 1000])

def pc_spectrum(hist_array, params=None, log=False):
    sns.set_style("white")
    plt.figure()
    bins = np.linspace(start=0, stop=max(hist_array), num=int(max(hist_array)))
    [n, bins, patches] = plt.hist(hist_array, bins=bins, edgecolor="none")
    plt.xlabel("ADC")
    plt.ylabel("Counts")
    if log:
        plt.yscale("log")
    if params:
        plt.plot(multi_gauss(bins, params), "r")

def plot_gain(sipm, lin_fit=False):
    plt.figure()
    plt.plot(sipm.bias, sipm.gain, '.')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Gain")

    if lin_fit:
        (slope, intercept) = linregress(x=sipm.bias, y=sipm.gain)
        x = np.linspace(sipm.bias[0], sipm.bias[-1], 100)
        y = np.multiply(slope, x)
        y = np.add(y, intercept)
        plt.plot(x, y, "r")
        plt.legend(["Breakdown Voltage: "+str(intercept)+" V"])
    plt.show()

def plot_dcr(sipm):
    plt.figure()
    plt.plot(sipm.bias, sipm.dark_rate)
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Dark Count Rate")
    plt.show()

def plot_cross_talk(sipm):
    plt.figure()
    plt.plot(sipm.bias, sipm.cross_talk, ".")
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Cross Talk Probability (%)")
    plt.show()

def plot_pde(sipm):
    plt.figure()
    plt.plot(sipm.bias, sipm.pde)
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Photon Detection Efficiency (%)")
    plt.show()









