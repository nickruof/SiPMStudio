import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from SiPMStudio.processing.functions import butter_bandpass
from SiPMStudio.processing.functions import multi_gauss

from scipy import fftpack
from scipy.signal import freqz
from scipy.stats import linregress
from scipy.stats import kde
from scipy.stats import expon
from scipy.signal import find_peaks
from functools import partial

sns.set_style("whitegrid")


def plot_hist(values_array, bins=500, x_range=None, density=False, labels=None):
    sns.set_style("ticks")
    colors = sns.color_palette()
    heights = []
    bin_edges = []
    all_patches = []
    for i, values in enumerate(values_array):
        [n, bins, patches] = plt.hist(values, bins=bins, range=x_range, edgecolor="none",
                                      color=colors[i], density=density, alpha=0.5)
        x_outline, y_outline = _outline_hist(n, bins)
        plt.plot(x_outline, y_outline, color=colors[i])
        heights.append(n)
        bin_edges.append(bins)
        all_patches.append(patches)
    plt.legend(labels)
    return heights, bin_edges, all_patches


def _outline_hist(heights, bin_edges):
    x_values = []
    y_values = []

    for i in range(len(bin_edges)-1):
        if i == 0:
            x_values.append(bin_edges[i])
            y_values.append(0)
        x_values.append(bin_edges[i])
        x_values.append(bin_edges[i+1])
        x_values.append(bin_edges[i+1])
        if i == len(bin_edges)-2:
            y_values.append(heights[i])
            y_values.append(heights[i])
            y_values.append(0)
        else:
            y_values.append(heights[i])
            y_values.append(heights[i])
            y_values.append(heights[i + 1])

    return np.array(x_values), np.array(y_values)


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
    plt.plot(x_fft/1.0e6, y_fft_norm)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (%)")
    plt.yscale("log")
    plt.show()


def plot_waveform(waveform, get_peaks=False, min_dist=None, min_height=None, width=0):
    time = np.linspace(0, 2*len(waveform)-1, len(waveform))
    plt.figure()
    plt.plot(time, waveform)
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (V)")

    if get_peaks:
        peaks, _properties = find_peaks(x=waveform, height=min_height, distance=min_dist, width=width)
        print(peaks)
        peak_heights = waveform.values[peaks]
        peak_times = time[peaks]
        plt.plot(peak_times, peak_heights, "r.")
    plt.show()


def waveform_plots(waveforms, get_peaks=False, min_dist=None, min_height=None, width=0):
    time = np.linspace(0, 2*waveforms.shape[1]-1, waveforms.shape[1])
    waveform_number = 0

    def key_event(event, fig):
        nonlocal waveform_number

        if event.key == "right":
            waveform_number = waveform_number + 1
        elif event.key == "left":
            waveform_number = waveform_number - 1
        else:
            return
        waveform_number = waveform_number % len(waveforms)
        plt.clf()
        plt.plot(time, waveforms.iloc[waveform_number, :])
        if get_peaks:
            peaks, _properties = find_peaks(waveforms.iloc[waveform_number, :],
                                            height=min_height, distance=min_dist, width=width)
            peak_heights = waveforms.iloc[waveform_number, :].values[peaks]
            peak_times = time[peaks]
            plt.plot(peak_times, peak_heights, "r.")
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", partial(key_event, fig=fig))
    plt.plot(time, waveforms.iloc[0, :])
    if get_peaks:
        peaks, _properties = find_peaks(waveforms.iloc[waveform_number, :],
                                        height=min_height, distance=min_dist, width=width)
        peak_heights = waveforms.iloc[0, :].values[peaks]
        peak_times = time[peaks]
        plt.plot(peak_times, peak_heights, "r.")


def plot_waveforms(waveforms):
    ax = self.waveforms.plot()
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (V)")
    ax.set_xlim([0, 1000])


def pc_spectrum(hist_array, n_bins=2000, log=False, density=False, labels=None):
    bins = np.linspace(start=min(hist_array[-1]), stop=max(hist_array[-1]), num=n_bins)
    [n, bins, patches] = sipm_plt.plot_hist(hist_array, bins, None, density, labels)
    plt.xlabel("ADC")
    if density:
        plt.ylabel("Norm Counts")
    else:
        plt.ylabel("Counts")
    if log:
        plt.yscale("log")


def ph_spectrum(heights_array, hist_range=None, log=False, density=False, labels=None):
    bins = 500
    [n, bins, patches] = sipm_plt.plot_hist(heights_array, bins, hist_range, density, labels)
    plt.xlabel("Pulse Heights (V)")
    plt.ylabel("Counts")
    if log:
        plt.yscale("log")


def plot_gain(sipm, lin_fit=False):
    plt.plot(sipm.bias, sipm.gain_magnitude, '.')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Gain")

    if lin_fit:
        (slope, intercept, _rvalue, _pvalue, _stderr) = linregress(x=sipm.bias, y=sipm.gain)
        x = np.linspace(sipm.bias[0], sipm.bias[-1], 100)
        y = np.multiply(slope, x)
        y = np.add(y, intercept)
        plt.plot(x, y, "r")
        plt.legend(["Breakdown Voltage: "+str(round(intercept, 1))+" V"])


def plot_dcr(sipm):
    plt.plot(sipm.bias, sipm.dark_rate)
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Dark Count Rate")


def plot_cross_talk(sipm):
    plt.plot(sipm.bias, sipm.cross_talk, ".")
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Cross Talk Probability (%)")


def plot_pde(sipm):
    plt.plot(sipm.bias, sipm.pde, ".")
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Photon Detection Efficiency (%)")


def plot_delay_times(dts, bins=500, bounds=None, fit=False, alpha=0.75):
    sns.set_style("ticks")
    if bounds is None:
        bounds = [0, 1e5]
    [n, bins, _patches] = plot_hist(dts, bins=bins, x_range=bounds, density=True)
    if fit:
        loc, scale = expon.fit(dts[(dts > bounds[0]) & (dts < bounds[1])])
        plt.plot(bins[0][:-1], expon.pdf(bins[0][:-1], loc=loc, scale=scale), color="r")
        plt.legend(["1/tau = "+str(round(1/(scale*1e-9)))+ " Hz"])
    plt.xlabel("Delay Times (ns)")
    plt.ylabel("Normalized Counts")
    plt.xscale("log")
    plt.yscale("log")


def plot_delay_height(dts, heights, density=False):
    if density:
        xy = np.vstack([dts, heights])
        z = kde.gaussian_kde(xy)(xy)
        plt.figure()
        plt.scatter(dts, heights, c=z, s=1)
        plt.colorbar()
    else:
        plt.figure()
        plt.scatter(dts, heights, c="b", s=1)
    plt.xscale("log")
    plt.xlabel("Delay Time (ns)")
    plt.ylabel("Pulse Heights (V)")
    plt.show()









