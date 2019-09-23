import numpy as np

from scipy.signal import butter


def line(x, m, b):
    return m*x + b


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


def exponential(x, a, tau):
    return a*np.exp(-x/tau)


def moyal(x, A, loc, scale):
    y = (x - loc) / scale
    return A*np.exp(-(y + np.exp(-y)) / 2)


def multi_gauss_moyal(x, A, loc, scale, *params):
    return multi_gauss(x, *params) + moyal(x, loc, scale)


def butter_bandpass(lowcut, highcut, sample_rate, order=5):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    (b, a) = butter(N=order, Wn=[low, high], btype="bandpass")
    return b, a


def sipm_rise(t, a, tau, t0, d):
    return a * (1 - np.exp(-(t-t0)/tau)) + d


def sipm_fall(t, a, tau, t0, d):
    return a * np.exp(-(t-t0)/tau) + d


def double_exp(t, A, lam1, lam2):
    distro_1 = A*np.exp(-lam1*t)
    distro_2 = (1 - A)*np.exp(-lam2*t)
    return distro_1 + distro_2

