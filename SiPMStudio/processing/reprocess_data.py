import os, time
import tqdm
import h5py
import numpy as np


def reprocess_data(settings, processor, verbose=False, chunk=2000, write_size=1):
    path_t2 = settings["output_path_t2"]
    output_files = []

    base_name = settings["file_base_name"]
    for entry in settings["init_info"]:
        bias_label = entry["bias"]
        output_files.append(f"t2_{base_name}_{bias_label}.h5")

    for idx, file in enumerate(output_files):
        destination = os.path.join(path_t2, file)
        h5_file = h5py.File(destination, "r")
        num_rows = h5_file["/raw/waveforms"][:].shape[0]
        df_storage = {}