import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings

from scipy.signal import find_peaks

import scipy.constants as const

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


def gain(peak_locs, peak_errors):
    peak_locs = np.array(peak_locs)
    peak_errors = np.array(peak_errors)
    peak_diffs = peak_locs[1:] - peak_locs[:-1]
    diff_errors = np.sqrt((peak_errors[1:] + peak_errors[:-1])**2)
    return np.sum(peak_diffs * diff_errors) / np.sum(diff_errors) / const.e


def all_dts(timetags, waveforms, dt, height=None, distance=None, width=None):
    all_times = []
    for i, wave in enumerate(tqdm.tqdm(waveforms, total=waveforms.shape[0])):
        peak_locs, heights = find_peaks(wave, height=height, distance=distance, width=width)
        times = [timetags[i] + dt*peak for peak in peak_locs]
        all_times.extend(times)
    return np.array(all_times[1:]) - np.array(all_times[:-1])


def amp_dt(timetags, waveforms, dt, norm_charges, trig_time=0, lower=0.5, height=None, distance=None, width=None, look_back=None):
    wf_dts = []
    wf_amps = []
    wf_ids = []

    signals = waveforms[norm_charges > lower]
    sig_times = timetags[norm_charges > lower]

    time_0 = -1
    time_1 = -1

    wf_idx = 0
    pbar = tqdm.tqdm(total=signals.shape[0])
    while wf_idx < signals.shape[0]:
        peak_locs, heights = find_peaks(signals[wf_idx], height=height, distance=distance, width=width)
        times = [sig_times[wf_idx] + dt*peak for peak in peak_locs]
        amps = None
        if look_back is not None:
            amps = []
            for peak in peak_locs:
                if peak-look_back > 0:
                    amps.append(signals[wf_idx][peak] - signals[wf_idx][peak-look_back])
                else:
                    amps.append(signals[wf_idx][peak])
        else:
            amps = [signals[wf_idx][peak] for peak in peak_locs]
        if len(times) == 0:
            wf_idx += 1
            pbar.update(1)
            continue
        diffs = list(np.absolute(np.array(times) - sig_times[wf_idx] - trig_time))
        idx = diffs.index(min(diffs))
        if len(times) == 1:
            if time_0 < 0:
                time_0 = times[0]
                wf_idx += 1
                pbar.update(1)
            elif time_0 > 0:
                time_1 = times[0]
                wf_dts.append(time_1 - time_0)
                wf_amps.append(amps[0])
                wf_ids.append(wf_idx)
                time_0 = -1
                time_1 = -1
                wf_idx += 1
                pbar.update(1)
        elif len(times) > idx+1:
            if time_0 < 0:
                time_0 = times[idx]
                time_1 = times[idx+1]
                wf_dts.append(time_1 - time_0)
                wf_amps.append(amps[idx+1])
                wf_ids.append(wf_idx)
                time_0 = -1
                time_1 = -1
                wf_idx += 1
                pbar.update(1)
            elif (time_0 > 0) & (time_1 < 0):
                time_1 = times[0]
                wf_dts.append(time_1 - time_0)
                wf_amps.append(amps[0])
                wf_ids.append(wf_idx)
                time_0 = -1
                time_1 = -1
                wf_idx += 1
                pbar.update(1)
        else:
            wf_idx += 1
            pbar.update(1)

    return np.array(wf_dts), np.array(wf_amps), np.array(wf_ids)


def cross_talk_frac(norm_charges, min_charge=0.5, max_charge=1.5):
    cross_events = (np.array(norm_charges)[norm_charges > max_charge]).shape[0]
    total_events = (np.array(norm_charges)[norm_charges > min_charge]).shape[0]

    error = np.sqrt((cross_events/total_events**2) + (cross_events**2/total_events**3))
    return cross_events / total_events, error


def excess_charge_factor(norm_charges, min_charge=0.5, max_charge=1.5):
    primary_charge = norm_charges[(norm_charges > min_charge) & (norm_charges < max_charge)]
    ecf = np.mean(norm_charges) / np.mean(primary_charge)
    return ecf
