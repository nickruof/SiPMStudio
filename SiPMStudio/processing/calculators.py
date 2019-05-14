import numpy as np

from SiPMStudio.io.file_settings import read_file


def normalize_e(path, file_name, params_data, wave_data=None):
    peak_locs = read_file(path, file_name)["peaks"]
    peak_locs = np.array(peak_locs)
    diffs = peak_locs[1:] - peak_locs[:-1]
    average_diff = np.mean(diffs)
    normalized_E = np.subtract(params_data["E_SHORT"].values, peak_locs[0])
    normalized_E = np.divide(normalized_E, average_diff)
    normalized_E = np.add(normalized_E, 1)
    return params_data.replace({"E_SHORT": normalized_E})








