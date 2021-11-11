import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings
import math

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

from SiPMStudio.processing.functions import gaussian
import SiPMStudio.plots.plots_base as plots_base
from SiPMStudio.processing.transforms import savgol

warnings.filterwarnings("ignore", "PeakPropertyWarning: some peaks have a width of 0")


def current_waveforms(waveforms, amp, vpp=2, n_bits=14):
    return waveforms * (vpp / 2 ** n_bits) / amp


def integrate_current(current_forms, lower_bound=0, upper_bound=200, sample_time=2e-9):
    return np.sum(current_forms.T[lower_bound:upper_bound].T, axis=1)*sample_time


def rando_integrate_current(current_forms, width, sample_time=2e-9):
    start_range = width
    stop_range = current_forms.shape[1] - width - 1
    start = np.random.randint(start_range, stop_range)
    stop = start + width
    return np.sum(current_forms.T[start:stop].T, axis=1)*sample_time


def cross_talk_frac(norm_charges, min_charge=0.5, max_charge=1.5):
    cross_events = np.array(norm_charges)[norm_charges > max_charge]
    total_events = np.array(norm_charges)[norm_charges > min_charge]
    return cross_events / total_events


def excess_charge_factor(norm_charges, min_charge=0.5, max_charge=1.5):
    primary_charge = norm_charges[(norm_charges > min_charge) & (norm_charges < max_charge)]
    ecf = np.mean(norm_charges) / np.mean(primary_charge)
    return ecf
