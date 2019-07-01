import numpy as np

from SiPMStudio.io.file_settings import read_file


def normalize_params(params_data, path, file_name, settings_option="pc_peaks", wave_data=None):

    # TODO: Remove dependency on CoMPASS variables

    peak_locs = np.array(read_file(path, file_name, file_type="waves")[settings_option])
    diffs = peak_locs[1:] - peak_locs[:-1]
    average_diff = np.mean(diffs)
    norm_e = np.subtract(params_data["E_SHORT"].values, peak_locs[0])
    norm_e = np.divide(norm_e, average_diff)
    norm_e = np.add(norm_e, 1)
    return params_data.replace({"E_SHORT": norm_e})




