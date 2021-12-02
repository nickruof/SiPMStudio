import numpy as np
import pandas as pd

from SiPMStudio.analysis.dark import current_waveforms, integrate_current

def charge(outputs, wf_in, out, window, amp):
    waveforms = outputs[wf_in]
    currents = current_waveforms(waveforms, outputs[amp])
    charges = integrate_current(currents, window[0], window[1])
    outputs[out] = charges


def normalize_charge(outputs, in_name, out_name, peak_locs, peak_errors):
    x0 = peak_locs[0]
    peak_locs = np.array(peak_locs)[1:]
    peak_errors = np.array(peak_errors)[1:]
    peak_diffs = peak_locs[1:] - peak_locs[:-1]
    diff_errors = np.sqrt((peak_errors[1:] + peak_errors[:-1])**2)
    gain = np.sum(peak_diffs * diff_errors) / np.sum(diff_errors)
    charges = outputs[in_name]
    outputs[out_name] = (charges - x0) / gain


def voltage_divider(R1, R2):
    return (R2 / (R1 + R2))


def trans_amp(R1):
    return - R1


def non_invert_amp(R1, R2):
    return (1 + (R1 / R2))


def invert_amp(R1, R2):
    return - R1 / R2
