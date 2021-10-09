import numpy as np
import pandas as pd

from SiPMStudio.analysis.dark import current_waveforms, integrate_current

def normalize_energy(params_data, pc_peaks, label, waves_data=None):
    diffs = pc_peaks[1:] - pc_peaks[:-1]
    average_diff = np.mean(diffs)
    norm_e = np.subtract(params_data[label].to_numpy(), pc_peaks[0])
    norm_e = np.divide(norm_e, average_diff)
    norm_e = np.add(norm_e, 1)
    output_params_data = pd.DataFrame(params_data.to_numpy(), index=params_data.index, columns=params_data.columns)
    output_params_data.loc[:, label] = norm_e
    return output_params_data


def normalize_charge(outputs, wf_in, out, gain, int_window, peak_locs=None):
    x0 = 0
    waveforms = outputs[wf_in]
    if peak_locs is None:
        x0 = peak_locs[0]

    current_forms = current_waveforms(waveforms)
    charges = integrate_current(current_forms, int_window[0], int_window[1])
    outputs[out] = (charges - x0) / gain
    



