import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import cross
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

def amp_dt(timetags, waveforms, dt, height=None, distance=None, width=None):
    wf_times = []
    wf_amps = []
    wf_ids = []

    wf_times_temp = []
    wf_amps_temp = []
    wf_ids_temp = []
    for i, wave in enumerate(tqdm.tqdm(waveforms, total=waveforms.shape[0])):
        peak_locs, heights = find_peaks(wave, height=height, distance=distance, width=width)
        times = [timetags[i] + dt*peak for peak in peak_locs]
        amps = [wave[peak] for peak in peak_locs]
        if (len(times) == 1) & (len(wf_times_temp) == 0):
            wf_times_temp.extend([times[0]])
            wf_amps_temp.extend([amps[0]])
            wf_ids_temp.extend([i])
        elif (len(times) >= 2) & (len(wf_times_temp) == 0):
            wf_times_temp.extend([times[0], times[1]])
            wf_amps_temp.extend([amps[0], amps[1]])
            wf_ids_temp.extend([i]*2)
        elif (len(times) > 0) & (len(wf_times_temp) == 1):
            wf_times_temp.extend([times[0]])
            wf_amps_temp.extend([amps[0]])
            wf_ids_temp.extend([i])
        if len(wf_times_temp) == 2:
            wf_times.extend(wf_times_temp)
            wf_amps.extend(wf_amps_temp)
            wf_ids.extend(wf_ids_temp)

            wf_times_temp.clear()
            wf_amps_temp.clear()
            wf_ids_temp.clear()

    wf_dts = np.array(wf_times)[1:] - np.array(wf_times)[:-1]
    wf_amps = np.array(wf_amps)[1:]
    wf_ids = np.array(wf_ids)[1:]
    return wf_dts, wf_amps, wf_ids


def cross_talk_frac(norm_charges, min_charge=0.5, max_charge=1.5):
    cross_events = (np.array(norm_charges)[norm_charges > max_charge]).shape[0]
    total_events = (np.array(norm_charges)[norm_charges > min_charge]).shape[0]

    error = np.sqrt((cross_events/total_events**2) + (cross_events**2/total_events**3))
    return cross_events / total_events, error


def excess_charge_factor(norm_charges, min_charge=0.5, max_charge=1.5):
    primary_charge = norm_charges[(norm_charges > min_charge) & (norm_charges < max_charge)]
    ecf = np.mean(norm_charges) / np.mean(primary_charge)
    return ecf
