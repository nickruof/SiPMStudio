import os, time
import tqdm
import h5py
import numpy as np

from SiPMStudio.processing.process_data import _chunk_range

def data_chunk(h5_file, begin, end):
    storage = {}
    for name in h5_file["/raw"].keys():
        storage[f"/raw/{name}"] = h5_file[f"/raw/{name}"][begin:end]

    for name in h5_file["/processed"].keys():
        storage[f"/processed/{name}"] = h5_file[f"/processed/{name}"][begin:end]
    return storage

def output_chunk(output, h5_file, begin, end):
    for key, value in output.items():
        h5_file[key][begin:end] = value


def reprocess_data(settings, processor, file_name=None, verbose=False, chunk=2000, write_size=1):
    path_t2 = settings["output_path_t2"]
    output_files = []

    if file_name is None:
        base_name = settings["file_base_name"]
        for entry in settings["init_info"]:
            bias_label = entry["bias"]
            output_files.append(f"t2_{base_name}_{bias_label}.h5")
    else:
        output_files.append(os.path.join(path_t2, file_name))

    if verbose:
        print(f"Files to reprocess: {output_files}")

    for idx, file in enumerate(output_files):
        destination = os.path.join(path_t2, file)
        if verbose:
            print(f"Reprocessing: {file}")
        h5_file = h5py.File(destination, "r+")
        num_rows = h5_file["/raw/timetag"][:].shape[0]
        for i in tqdm.tqdm(range(num_rows//chunk + 1)):
            begin, end = _chunk_range(i, chunk, num_rows)
            if (end - num_rows) < chunk:
                end = num_rows - 1
            storage = data_chunk(h5_file, begin, end)
            output_storage = _process_chunk(storage, processor)
            output_chunk(output_storage, h5_file, begin, end)
            processor.reset_outputs()
        h5_file.close()


def _process_chunk(storage, processor):
    processor.init_outputs(storage)
    return processor.process()
