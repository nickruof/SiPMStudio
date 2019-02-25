import numpy as np
import pandas as pd

from .core import digitizers

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