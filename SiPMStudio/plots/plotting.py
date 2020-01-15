import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from SiPMStudio.processing.functions import butter_bandpass
from SiPMStudio.plots import plots_base

from uncertainties import unumpy
from scipy import fftpack
from scipy.signal import freqz
from scipy.stats import linregress
from scipy.stats import kde
from scipy.stats import expon
from scipy.signal import find_peaks
from functools import partial

sns.set_style("ticks")


def plot_butter_response(ax, digitizer, lowcut, highcut, order=5):
    (b, a) = butter_bandpass(lowcut, highcut, digitizer.sample_rate, order=order)
    (w, h) = freqz(b, a, worN=2000)
    ax.plot(((fs * 0.5)/np.pi) * w, abs(h), label="order = %d" % order)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain")


def plot_fft(ax, digitizer, waveform):
    wave_fft = fftpack.fft(waveform)
    N = len(waveform)
    sample_period = 1 / digitizer.sample_rate
    x_fft = np.linspace(0.0, 1.0/(2.0*sample_period), N/2)
    y_fft = 2.0/N * np.abs(wave_fft[:N//2])
    fft_norm = np.linalg.norm(y_fft)
    y_fft_norm = [element/fft_norm for element in y_fft]
    ax.plot(x_fft/1.0e6, y_fft_norm)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (%)")
    ax.set_yscale("log")


def plot_waveform(ax, waveform, get_peaks=False, min_dist=None, min_height=None, width=0):
    time = np.linspace(0, 2*len(waveform)-1, len(waveform))
    ax.plot(time, waveform)
    ax.xlabel("Time (ns)")
    ax.ylabel("Voltage (V)")

    if get_peaks:
        peaks, _properties = find_peaks(x=waveform, height=min_height, distance=min_dist, width=width)
        print(peaks)
        peak_heights = waveform.values[peaks]
        peak_times = time[peaks]
        ax.plot(peak_times, peak_heights, "r+")


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
            plt.plot(peak_times, peak_heights, "r+")
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", partial(key_event, fig=fig))
    plt.plot(time, waveforms.iloc[0, :])
    if get_peaks:
        peaks, _properties = find_peaks(waveforms.iloc[waveform_number, :],
                                        height=min_height, distance=min_dist, width=width)
        peak_heights = waveforms.iloc[0, :].values[peaks]
        peak_times = time[peaks]
        plt.plot(peak_times, peak_heights, "r+")


def height_scan(waveforms, bins):
    index_number = 0

    def key_event(event, figure, axes):
        nonlocal index_number

        if event.key == "right":
            index_number = index_number + 1
        elif event.key == "left":
            index_number = index_number - 1
            if index_number < 0:
                index_number = 0
        else:
            return
        plt.clf()
        plots_base.plot_hist(axes, [waveforms.iloc[:, index_number]], bins=bins)
        axes.set_xlabel("Waveform Number")
        axes.set_title("Waveform Index: "+str(index_number))
        figure.canvas.draw()

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("key_press_event", partial(key_event, figure=fig, axes=ax))
    plots_base.plot_hist(ax, [waveforms.iloc[:, 0]], bins=bins)
    ax.set_xlabel("Waveform Number")
    ax.set_title("Waveform Index: " + str(index_number))


def plot_waveforms(ax, waveforms, linewidth=0.01):
    times = np.repeat([range(0, 2*waveforms.shape[1], 2)], waveforms.shape[0], axis=0)
    ax.plot(times.T, waveforms.to_numpy().T, color=sns.color_palette()[0], linewidth=linewidth, alpha=0.075)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (V)")
    ax.set_xlim([20, 150])
    ax.set_ylim([np.amin(waveforms.to_numpy()), np.amax(waveforms.to_numpy())])


def pc_spectrum(ax, hist_array, n_bins=120, log=False, density=False, labels=None):
    bins = np.linspace(start=min(hist_array[-1]), stop=max(hist_array[-1]), num=n_bins)
    [n, bins, patches] = plots_base.plot_hist(ax, hist_array, bins, None, density, labels)
    plt.xlabel("ADC")
    if density:
        ax.set_ylabel("Norm Counts")
    else:
        ax.set_ylabel("Counts")
    if log:
        ax.set_yscale("log")
    return n, bins


def ph_spectrum(ax, heights_array, n_bins=500, hist_range=None, log=False, density=False, labels=None):
    [n, bins, patches] = plots_base.plot_hist(ax, heights_array, n_bins, hist_range, density, labels)
    ax.set_xlabel("Pulse Heights (V)")
    ax.set_ylabel("Counts")
    if log:
        ax.set_yscale("log")
    return n, bins


def snr(ax, sipm):
    noise_power = np.array(sipm.noise_power)
    signal_power = np.array(sipm.signal_power).T
    labels = []
    for i, power in enumerate(signal_power):
        inner_term = np.divide(power - noise_power[i], noise_power[i])
        signal_to_noise = 10 * np.log10(inner_term)
        plots_base.line_plot(ax, sipm.bias, signal_to_noise)
        labels.append(str(i+1)+" "+"p.e. peak")
    ax.legend(labels)
    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("SNR (dB)")


def gain(ax, sipm, lin_fit=False, save_path=None):
    plots_base.error_plot(ax, sipm.bias, sipm.gain_magnitude)
    ax.set_xlabel("Bias Voltage (V)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Gain")

    if lin_fit:
        (slope, intercept, _rvalue, _pvalue, _stderr) = linregress(x=sipm.bias, y=unumpy.nominal_values(sipm.gain_magnitude))
        x = np.linspace(sipm.bias[0], sipm.bias[-1], 100)
        y = np.multiply(slope, x)
        y = np.add(y, intercept)
        ax.plot(x, y, "r")
        ax.legend(["Breakdown Voltage: " + str(round(-intercept/slope, 1)) + " V"])

    if save_path:
        plt.savefig(save_path+"/gain_plot.jpg")


def dcr(ax, sipm, save_path=None):
    dark_count_rate = [dr / 1000 / sipm.area for dr in sipm.dcr_fit]
    plots_base.error_plot(ax, sipm.bias, dark_count_rate)
    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("Dark Count Rate (kHz/mm^2)")

    if save_path:
        plt.savefig(save_path+"/dcr_plot.jpg")


def cross_talk(ax, sipm, save_path=None):
    plots_base.error_plot(ax, sipm.bias, np.array(sipm.cross_talk)*100)
    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("Cross Talk Probability (%)")

    if save_path:
        plt.savefig(save_path+"/cross_talk_plot.jpg")


def delay_times(ax, dts, bins=500, bounds=None, fit=False, alpha=0.75):
    if bounds is None:
        bounds = [0, 1e5]
    [n, bins, _patches] = plots_base.plot_hist(ax, [dts], bins=bins, x_range=bounds, density=True)
    if fit:
        loc, scale = expon.fit(dts[(dts > bounds[0]) & (dts < bounds[1])])
        red_line = ax.plot(bins[0][:-1], expon.pdf(bins[0][:-1], loc=loc, scale=scale), color="r")
        ax.legend(red_line, "1/tau = "+str(round(1/(scale*1e-9))) + " Hz")
    ax.set_xlabel("Delay Times (ns)")
    ax.set_ylabel("Normalized Counts")
    ax.set_xscale("log")
    ax.set_yscale("log")


def delay_heights(fig, ax, dts, heights, density=False):
    if density:
        xy = np.vstack([dts, heights])
        z = kde.gaussian_kde(xy)(xy)
        ax.scatter(dts, heights, c=z, s=1)
        fig.colorbar()
    else:
        ax.scatter(dts, heights, c="b", s=1)
    ax.set_xscale("log")
    ax.set_xlabel("Delay Time (ns)")
    ax.set_ylabel("Pulse Heights (V)")


def pde(ax, sipm, save_path=None):
    pde_percent = [decimal*100 for decimal in sipm.pde]
    plots_base.error_plot(ax, sipm.bias, pde_percent)
    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("Photon Detection Efficiency (%)")

    if save_path:
        plt.savefig(save_path+"/pde_plot.jpg")