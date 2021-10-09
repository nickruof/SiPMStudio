import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings
import math

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

from SiPMStudio.processing.functions import gaussian, rise_func
import SiPMStudio.plots.plots_base as plots_base
from SiPMStudio.processing.transforms import savgol

warnings.filterwarnings("ignore", "PeakPropertyWarning: some peaks have a width of 0")


def current_waveforms(waveforms, vpp=2, n_bits=14):
    return waveforms * (vpp / 2 ** n_bits) * (1000 / 31.05) * 1.0e-6


def integrate_current(current_forms, lower_bound=0, upper_bound=200, sample_time=2e-9):
    return np.sum(current_forms.T[lower_bound:upper_bound].T, axis=1)*sample_time


def rando_integrate_current(current_forms, width, sample_time=2e-9):
    start_range = width
    stop_range = current_forms.shape[1] - width - 1
    start = np.random.randint(start_range, stop_range)
    stop = start + width
    return np.sum(current_forms.T[start:stop].T, axis=1)*sample_time


def integrate_slices(current_forms, width, window_size, sample_time=2e-9):
    steps = math.floor((current_forms.shape[1] - 2*window_size) / window_size)
    charges = []
    for i in range(steps):
        start = window_size * (i+1)
        stop = start + width
        charges += list(np.sum(current_forms.T[start:stop].T, axis=1)*sample_time)
    return np.array(charges)


def wave_peaks(waveforms, height=500, distance=5):
    all_peaks = []
    all_heights = []
    for waveform in tqdm.tqdm(waveforms, total=len(waveforms)):
        peak_locs = find_peaks(waveform, height=height, distance=distance)[0]
        heights = waveform[peak_locs]
        if len(heights) == len(peak_locs):
            all_peaks.append(peak_locs)
            all_heights.append(heights)
    return np.asarray(all_peaks, dtype=object), np.asarray(all_heights, dtype=object)


def cross_talk_frac(heights, min_height=0.5, max_height=1.50):
    one_pulses = 0
    other_pulses = 0
    for height_set in tqdm.tqdm(heights, total=len(heights)):
        if len(height_set) > 0:
            if (height_set[0] > min_height) & (height_set[0] < max_height):
                one_pulses += 1
            else:
                other_pulses += 1
        else:
            continue
    return other_pulses / (one_pulses + other_pulses)


def cross_talk_frac_v2(peaks, peak_errors, charges):
    charge_diff = 0
    if any(np.isnan(peak_errors)) | any(np.isinf(peak_errors)):
        charge_diff = peaks[1] - peaks[0]
    else:
        diffs = peaks[1:] - peaks[:-1]
        errors_squared = peak_errors**2
        errors = np.sqrt(errors_squared[1:] + errors_squared[:-1])
        weights = 1 / errors
        charge_diff = np.average(diffs, weights=weights)
    one_charges = charges[(charges > (charge_diff/2)) & (charges < (3*charge_diff/2))]
    other_charges = charges[charges > (3*charge_diff/2)]
    return len(one_charges) / (len(one_charges) + len(other_charges))


def excess_charge_factor(norm_charges, min_charge=0.5, max_charge=1.5):
    primary_charge = norm_charges[(norm_charges > min_charge) & (norm_charges < max_charge)]
    ecf = np.mean(norm_charges) / np.mean(primary_charge)
    return ecf
