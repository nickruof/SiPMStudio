import numpy as np

from SiPMStudio.calculations.helpers import detect_peaks

from scipy.optimize import curve_fit


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


def butter_bandpass(digitizer, lowcut, highcut, order=5):
    nyq = 0.5 * digitizer.sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    (b, a) = butter(order, [low, high], btype="bandpass")
    return (b, a)


def fit_multi_gauss(bins, bin_vals, min_dist, min_height, display=False):
    peaks = detect_peaks(bin_vals, mpd=min_dist, mph=min_height)
    amplitudes = bin_vals[peaks]
    sigmas = [17]*len(peaks) #method needed to avoid hard coded sigma guess
    guess = []
    for i, peak in enumerate(peaks):
        guess.append(peak)
        guess.append(amplitudes[i])
        guess.append(sigmas[i])
    (popt, pcov) = curve_fit(multi_gauss, xdata=bins[:-1], ydata=bin_vals, p0=guess)
    fit = multi_gauss(bins[:-1], *popt)
    if display:
        plt.figure()
        plt.plot(bins[:-1], fit, color="red")
    return popt