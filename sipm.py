""" SiPM Characterization software compatible with the CAEN CoMPASS readout system
    with output in csv format
    Nicholas Ruof University of Washington CENPA """

from pint import UnitRegistry
from scipy import fftpack
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from scipy.signal import freqz
from scipy.sparse import diags
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from statsmodels.robust import mad

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns

from detect_peaks import detect_peaks

sns.set()

#### Global Variables ###################################################################################
#Units Library
ureg = UnitRegistry()

# Digitizer Settings

SAMPLE_RATE = 500.0e6 # Samples / Second
V_RANGE = 2
N_BITS = 14
ECAL = 2.5e-15 #C/LSB

#Resolution estimate of p.e. peaks in E_short histogram
E_SIGMA = 17

#### Helper Functions ###################################################################################
def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x-mu)**2/(2*sigma**2))

def multi_gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + gaussian(x, ctr, wid, amp)
    return y

def butter_bandpass(lowcut, highcut, fs=SAMPLE_RATE, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    (b, a) = butter(order, [low, high], btype="bandpass")
    return (b, a)

def plot_butter_response(lowcut, highcut, fs=SAMPLE_RATE, order=5):
    (b, a) = butter_bandpass(lowcut, highcut, fs, order=order)
    (w, h) = freqz(b, a, worN=2000)
    plt.figure()
    plt.plot(((fs * 0.5)/np.pi) * w, abs(h), label="order = %d" % order)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")

#### SiPM Class #########################################################################################
class sipm:

    def __init__(self, path_name):
        self.path = path_name
        self.params_frame = pd.DataFrame() # timetag, E_short, E_long
        self.waveforms = pd.DataFrame()
        self.processed_waveforms = []
        self.index = 0
        self.shift_value = False

        self.n = []
        self.bins = []
        self.params = []
        self.delay_times = []
        self.heights = []

        self.gain = 1.0
        self.cross_talk = 0
        self.afterpulsing = 0

        # For plotting IV curve
        self.current = []
        self.bias = []

    def load_csv_file(self, file_name, waves=False):
        print("Loading "+file_name+" ...")
        csv = pd.read_csv(self.path+file_name, delimiter=";", header=None, skiprows=1)
        if waves:
            self.params_frame = csv.iloc[:, :3]
            self.params_frame.columns = ["timetag", "E_short", "E_long"]
            self.waveforms = csv.iloc[:, 4:].copy()
            self.waveforms = self.waveforms.transpose().set_index(keys=np.array(range(0, 2*self.waveforms.shape[1], 2)))
            self.processed_waveforms = self.waveforms.copy()
        else:
            params = csv
            self.params_frame = pd.DataFrame(params, columns=["timetag", "E_short", "E_long"])
        self.bins = [i for i in range(int(max(self.params_frame["E_long"])))]
        print("Finished!")
    #Waveform Filters and Transforms////////////////////////////////////////////////////////////////////////////////////////////////////////

    def shift_all(self, simple=True):
        print("Shifting Waveforms ...")
        self.processed_waveforms = self.processed_waveforms - self.processed_waveforms.mean()
        self.processed_waveforms = np.multiply((V_RANGE/2**N_BITS), self.processed_waveforms)
        self.shift_value = True
        print("Finished!")

    def reset_waveforms(self):
        self.processed_waveforms = self.waveforms.copy()

    def savgol_waveforms(self, window=15, order=2):
        print("Saviztky-Golay")
        filtered_data = savgol_filter(self.processed_waveforms.values, window, order, axis=0)
        self.processed_waveforms = pd.DataFrame(data=filtered_data, index=self.waveforms.index, columns=self.waveforms.columns)
        print("Done!")

    def butter_bandpass_filter(self, lowcut, highcut, fs=SAMPLE_RATE, order=5):
        print("Butterworth BandPass Filter")
        (b, a) = butter_bandpass(lowcut, highcut, fs, order=order)
        filtered_data = filtfilt(b, a, self.processed_waveforms.values, axis=0)
        self.processed_waveforms = pd.DataFrame(data=filtered_data, index=self.waveforms.index, columns=self.waveforms.columns)
        print("Done!")

    def wavelet_denoise(self, wavelet="db2", levels=1, mode="soft"):
        print("Wavelet Denoise")
        data = self.processed_waveforms.values
        coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=levels, axis=0)
        sigma = mad(coeffs[-levels], axis=0)
        uthresh = sigma * np.sqrt(2 * np.log(data.shape[0]))
        coeffs[1:] = (pywt.threshold(coeffs[i], value=uthresh, mode=mode)  for i in range(1, len(coeffs)))
        filtered_data = pywt.waverec(coeffs, wavelet, axis=0)
        self.processed_waveforms = pd.DataFrame(data=filtered_data, index=self.waveforms.index, columns=self.waveforms.columns)
        print("Done!")

    def plot_FFT(self, wave_number=0, filtered=False):
        if filtered:
            wave_fft = fftpack.fft(self.processed_waveforms[wave_number])
        else:
            wave_fft = fftpack.fft(self.waveforms[wave_number])
        N = len(self.waveforms[wave_number])
        sample_period = 1 / SAMPLE_RATE
        x_fft = np.linspace(0.0, 1.0/(2.0*sample_period), N/2)
        y_fft = 2.0/N * np.abs(wave_fft[:N//2])
        fft_norm = np.linalg.norm(y_fft)
        y_fft_norm = [element/fft_norm for element in y_fft]
        plt.figure()
        plt.plot(x_fft, y_fft_norm)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (%)")

    #/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def time_interval(self):
        interval = self.params_frame["timetag"].iloc[-1] - self.params_frame["timetag"].iloc[0]
        interval = interval * 1.0e-12 * ureg.second
        return interval

    def waveblock(self, low, high, processed=True):
        if not processed:
            if high is None:
                wave_data = self.waveforms.iloc[:, low]
            else:
                wave_data = self.waveforms.iloc[:, low:high]
        else:
            if high is None:
                wave_data = self.processed_waveforms.iloc[:, low]
            else:
                wave_data = self.processed_waveforms.iloc[:, low:high]
        return wave_data



    def plot_waveform(self, wave_number, option="b-", find_peaks=False, min_height=0.0, min_distance=20.0, thresh=0, filtered=False):
        if not self.waveforms.empty:
            plt.figure()
            time = np.linspace(0, 2*self.waveforms.shape[0]-1, self.waveforms.shape[0])
            if filtered:
                plt.plot(time, self.processed_waveforms[wave_number], option)
                plt.ylabel("V")
            else:
                plt.plot(time, self.waveforms[wave_number], option)
                plt.ylabel("Raw ADC")
            plt.xlabel("Time (ns)")
            plt.title("Waveform "+str(wave_number)+" of "+str(self.waveforms.shape[1]-1))

            if find_peaks:
                if filtered:
                    peaks = detect_peaks(self.processed_waveforms[wave_number], mph=min_height, mpd=min_distance)
                    peak_heights = self.processed_waveforms.iloc[peaks, wave_number]
                    peak_times = time[peaks]
                    plt.plot(peak_times, peak_heights, "r.")
                else:
                    peaks = detect_peaks(self.waveforms[wave_number], mph=min_height, mpd=min_distance, threshold=thresh)
                    peak_heights = self.waveforms.iloc[peaks, wave_number]
                    peak_times = time[peaks]
                    plt.plot(peak_times, peak_heights, "r.")
        else:
            print("No Waveforms Loaded!")

    def plot_next(self):
        if self.waveforms:
            plt.figure()
            time = np.linspace(0, len(self.waveforms[0]), int(len(self.waveforms)/2))
            plt.plot(time, self.waveforms[self.index])
            plt.xlabel("Time (ns)")
            plt.title("Waveform "+str(self.index)+" of "+str(self.waveforms.shape[1]-1))
        if self.shift_value:
            plt.ylabel("V")
        else:
            plt.ylabel("Raw ADC")

        if self.index < len(self.waveforms)-1:
            self.index = self.index + 1
        else:
            print("No Waveforms Loaded!")

    def reset_index(self, number=0):
        if number < len(self.waveforms):
            self.index = number
        else:
            print("Index out of range!")


    def plot_all(self):
        ax = self.waveforms.plot()
        ax.set_xlabel("Time (ns)")
        if self.shift_value:
            ax.set_ylabel("V")
        else:
            ax.set_ylabel("Raw ADC")

    def integrate(self, window, processed=True):
        if processed:
            return np.multiply(np.sum(self.processed_waveforms.iloc[0:window, :].values, axis=0), 1/SAMPLE_RATE)
        else:
            return np.multiply(np.sum(self.waveforms.iloc[0:window, :].values, axis=0), 1/SAMPLE_RATE)



    def PC_spectrum(self, xbounds, errors=False, display=True, log=False, save=False):
        fig = plt.figure()
        if not errors:
            [n, bins, patches] = plt.hist(self.params_frame["E_long"], edgecolor="none", bins=self.bins)
            self.n = n
        else:
            [n, bins, patches] = plt.hist(self.params_frame["E_long"], bins=self.bins)
            self.n = n
            widths = bins[1] - bins[0]
            centers = (bins[:-1]+bins[1:]) / 2
            yerrs = []
        #90% confidence intervals
            for i, value in enumerate(n):
                if np.sqrt(value) > 0:
                    yerrs.append((np.std(value) / np.sqrt(value))*1.645)
                else:
                    yerrs.append(0.0)
            plt.errorbar(centers, n, yerrs, widths, marker=".", linestyle="", color="r")
        if xbounds:
            plt.xlim(xbounds[0], xbounds[1])
        plt.xlabel("ADC")
        plt.ylabel("Counts")
        if log:
            plt.yscale("log")
        plt.title("Pulse Charge Spectrum")
        if save:
            plt.savefig(self.path+"_pc_spectrum.png")
        if not display:
            plt.close(fig)
        if not errors:
            return (n, bins)
        else:
            return (n, bins, yerrs)

    def fit_multi_gauss_spectrum(self, min_distance, min_height):
        peaks = detect_peaks(self.n, mpd=min_distance, mph=min_height)
        amplitudes = self.n[peaks]
        sigmas = [E_SIGMA]*len(peaks)
        guess = []
        for i, peak in enumerate(peaks):
            guess.append(peak)
            guess.append(amplitudes[i])
            guess.append(sigmas[i])
        (popt, pcov) = curve_fit(multi_gauss, xdata=self.bins[:-1], ydata=self.n, p0=guess)
        fit = multi_gauss(self.bins[:-1], *popt)
        plt.figure()
        plt.plot(self.bins[:-1], fit, color="red")
        self.params = popt
        return popt

    def compute_gain(self, convert=False, sum_len=1):
        diffs = []
        gain_average = 1
        for i in range(0, len(self.params)-3, 3):
            diffs.append(self.params[i+3] - self.params[i])
        gain_average = sum(diffs[0:sum_len]) / float(len(diffs[0:sum_len]))
        self.gain = gain_average
        if convert:
            gain_average = gain_average * ECAL/1.6e-19
        return gain_average

    def dark_count_rate(self, units="hz"):
        index = int(self.params[0] - (self.gain/2))
        total_counts = sum(self.n[index:])
        dark_rate = (total_counts/self.time_interval()).to("Hz")
        if units == "khz":
            dark_rate.ito(ureg.kilohertz)
        elif units == "mhz":
            dark_rate.ito(ureg.megahertz)
        return dark_rate

    def cross_talk_prob(self):
        index1 = int(self.params[0] - (self.gain/2))
        index2 = int(self.params[3] - (self.gain/2))
        total_counts1 = sum(self.n[index1:])
        total_counts2 = sum(self.n[index2:])
        prob = total_counts2 / total_counts1
        return prob

    def delay_time_vs_height(self, min_height, min_distance, display=True):
        all_dts = []
        all_heights = []
        all_times = []

        for i in range(0, self.processed_waveforms.shape[1]):
            peaks = detect_peaks(self.processed_waveforms[i], mph=min_height, mpd=min_distance)
            times = np.add(self.params_frame.iloc[i, 0]*10**-3, 2*peaks)
            heights = self.processed_waveforms.iloc[peaks, i].values
            all_times = np.append(all_times, [times])
            all_heights = np.append(all_heights, [heights])
        M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
        all_dts = M_diag @ all_times
        all_dts = np.delete(all_dts, -1)
        all_heights = np.delete(all_heights, -1)

        if display:
            plt.figure()
            plt.scatter(all_dts, all_heights, c="b", s=1)
            plt.xscale("log")
            plt.xlabel("Delay Time (ns)")
            plt.ylabel("Pulse Heights (V)")
        self.delay_times = all_dts
        self.heights = all_heights

    def delay_times(self):
        bins = [i for i in range(int(max(self.delay_times)))]
        plt.figure()
        plt.hist(self.delay_times, bins=bins, edgecolor="none")
        plt.xlabel("Delay Times (ns)")
        plt.ylabel("Counts")
        plt.xscale("log")
        plt.yscale("log")

    def pulse_heights(self, log=False):
        bins = [i for i in range(int(max(self.heights)))]
        plt.figure()
        plt.hist(self.heights, bins=bins, edgecolor="none")
        plt.xlabel("Heights (V)")
        plt.ylabel("Counts")
        if log:
            plt.yscale("log")

    def gaussian_mixture(self, n_clusters):
        data = np.array([self.delay_times, self.heights])
        features = pd.DataFrame(data.T, columns=["dts", "heights"])
        gauss_mix = GaussianMixture(n_components=n_clusters).fit(features)
        features["labels"] = gauss_mix.predict(features)
        x = np.linspace(0, max(self.delay_times))
        y = np.linspace(0, max(self.heights))
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = - gauss_mix.score_samples(XX)
        Z = Z.reshape(X.shape)

        plt.figure()
        #plt.contourf(X, Y, Z, norm = colors.LogNorm(vmin=1.0, vmax=1000.0), levels = np.logspace(0, 3, 10), cmap = "Greens")
        plt.scatter(self.delay_times, self.heights, s=1, c=features["labels"])
        plt.xscale("log")

        return features["labels"]

    def dbscan(self, max_dist, min_samples):
        data = np.array([self.delay_times, self.heights])
        features = pd.DataFrame(data.T, columns=["dts", "heights"])
        dbscan = DBSCAN(eps=max_dist, min_samples=min_samples).fit(features)
        features["labels"] = dbscan.labels_

        plt.figure()
        plt.scatter(self.delay_times, self.heights, s=1, c=features["labels"])
        plt.xscale("log")

        n_clusters = len(set(features["labels"])) - (1 if -1 in features["labels"] else 0)
        n_noise = list(features["labels"]).count(-1)
        return (n_clusters, n_noise)