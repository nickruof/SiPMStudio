import os, time
import h5py
import numpy as np

from SiPMStudio.processing.process_data import _chunk_range
from SiPMStudio.utils.gen_utils import tqdm_range

def data_chunk(h5_file, begin, end):
    storage = {}
    for name in h5_file["/raw"].keys():
        if len(h5_file[f"/raw/{name}"].shape) > 0:
            storage[f"/raw/{name}"] = h5_file[f"/raw/{name}"][begin:end]

    for name in h5_file["/processed"].keys():
        if len(h5_file[f"/processed/{name}"].shape) > 0:
            storage[f"/processed/{name}"] = h5_file[f"/processed/{name}"][begin:end]
    return storage

def output_chunk(output, h5_file, begin, end):
    data_len = h5_file["/raw/timetag"].shape[0]
    for key, value in output.items():
        if key in h5_file:
            if value.shape[0] == data_len:
                h5_file[key][begin:end] = value
            else:
                h5_file[key].resize(h5_file[key].shape[0]+value.shape[0], axis=0)
                h5_file[key][-value.shape[0]:] = value
        elif len(value.shape) == 2:
            h5_file.create_dataset(key, data=value, maxshape=(None, None))
        elif len(value.shape) == 1:
            h5_file.create_dataset(key, data=value, maxshape=(None,))
        else:
            raise ProcessLookupError(f"Unable to create or add to dataset {key}")


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
        with h5py.File(destination, "r+") as h5_file:
            #_output_date(h5_file, "reprocess_date")
            num_rows = h5_file["/raw/timetag"][:].shape[0]
            for i in tqdm_range(0, num_rows//chunk + 1, verbose=verbose):
                begin, end = _chunk_range(i, chunk, num_rows)
                storage = data_chunk(h5_file, begin, end)
                output_storage = _process_chunk(storage, processor)
                output_chunk(output_storage, h5_file, begin, end)
                processor.reset_outputs()


def _process_chunk(storage, processor):
    processor.init_outputs(storage)
    return processor.process()


def _output_date(output_file, label=None):
    if label is None:
            label = "date"
    if label not in output_file:
        output_file.create_dataset(label, data=int(time.time()))
    else:
        output_file[label] = int(time.time())
