import numpy as np


def normalize_energy(params_data, pc_peaks, label, waves_data=None):
    diffs = pc_peaks[1:] - pc_peaks[:-1]
    average_diff = np.mean(diffs)
    norm_e = np.subtract(params_data[label].to_numpy(), pc_peaks[0])
    norm_e = np.divide(norm_e, average_diff)
    norm_e = np.add(norm_e, 1)
    return params_data.replace({label: norm_e})




