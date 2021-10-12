import numpy as np
import pandas as pd

from SiPMStudio.analysis.dark import current_waveforms, integrate_current

def charge(outputs, wf_in, out, window):
    waveforms = outputs[wf_in]
    currents = current_waveforms(waveforms)
    charges = integrate_current(currents, window[0], window[1])
    outputs[out] = charges


def normalize_charge(outputs, in_name, out_name, peak_locs, peak_errors):
    gain = np.sum(peak_locs * peak_errors) / np.sum(peak_errors)
    x0 = peak_locs[0]
    charges = outputs[in_name]
    outputs[out_name] = (charges - x0) / gain
    



