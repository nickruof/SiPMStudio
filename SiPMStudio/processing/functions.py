import numpy as np

from scipy.signal import butter


def line(x, m, b):
    return m*x + b


def quadratic(x, a, b, c):
    return a*x**2 + b*x + c


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


def butter_bandpass(lowcut, highcut, sample_rate, order=5):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    (b, a) = butter(N=order, Wn=[low, high], btype="bandpass")
    return b, a


def exp_func(t, A, t0, tau, D):
    return A*np.exp(-(t-t0)/tau) + D


def double_exp(t, A1, A2, t0, tau1, tau2, D):
    return exp_func(t, A1, t0, tau1, 0) + exp_func(t, A2, t0, tau2, D)


def rise_func(t, A, t0, tau, D):
    return A*(1 - np.exp(-(t-t0)/tau)) + D