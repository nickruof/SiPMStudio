import numpy as np

from scipy.signal import butter

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


def moyal(x, A, loc, scale):
    y = (x - loc) / scale
    return A*np.exp(-(y + np.exp(-y)) / 2)


def multi_gauss_moyal(x, A, loc, scale, *params):
    return multi_gauss(x, *params) + moyal(x, loc, scale)


def butter_bandpass(digitizer, lowcut, highcut, order=5):
    nyq = 0.5 * digitizer.sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    (b, a) = butter(N=order, Wn=[low, high], btype="bandpass")
    return b, a