import os, time
import h5py
import numpy as np

from SiPMStudio.utils.gen_utils import tqdm_range
from SiPMStudio.processing.process_data import _copy_to_t2, _initialize_outputs

def process_data(comm, rank, chunk_idx, processor, h5_input, h5_output, bias=None, overwrite=False, verbose=False, chunk=2000, write_size=1):

    start = time.time()
    # -----Processing Begins Here!---------------------------------
    num_rows = h5_input["n_events"][()]
    data_storage = {"size": 0}
    write_count = 0
    write_begin = 0
    write_end = 0

    for i in tqdm_range(chunk_idx[0], chunk_idx[1], text=f"Proc: {rank}", position=rank, verbose=verbose):
        write_count += 1
        begin, end = _chunk_range(i, chunk, num_rows)
        if write_count == 1:
            write_begin = begin
            write_end = end
        else:
            write_end += chunk
        output_data = _process_chunk(h5_input, processor, begin, end)
        _output_chunk(h5_output, output_data, data_storage, write_size, num_rows, chunk, write_begin, write_end)
        processor.reset_outputs()
        if write_count == write_size:
            write_count = 0


def _chunk_range(index, chunk, num_rows):
    start = index * chunk
    stop = (index+1) * chunk
    if stop >= num_rows:
        stop = num_rows
    return start, stop


def _process_chunk(h5_input, processor, begin, end):
    for key in h5_input["/raw/channels"]:
        processor.add_output(f"/raw/channels/{key}/waveforms",
                             h5_input[f"/raw/channels/{key}/waveforms"][begin: end])
    processor.add_output("timetag", h5_input["timetag"][begin: end])
    return processor.process()


def _output_chunk(output_file, chunk_data, storage, write_size, num_rows, chunk, start, stop):
    output_to_file = False
    if (write_size == 1) | (num_rows < chunk):
        output_to_file = True
    elif stop >= num_rows:
        output_to_file = True
    elif storage["size"] == (write_size - 1):
        output_to_file = True

    for i, output in enumerate(chunk_data.keys()):
        if output not in storage:
            storage[output] = []
        storage[output].append(chunk_data[output])
        if i == 0:
            storage["size"] = len(storage[output])
        if output_to_file:
            storage[output] = np.concatenate(storage[output])
    if output_to_file:
        _output_to_file(output_file, storage, start, stop)
        storage.clear()
        storage["size"] = 0


def _output_to_file(output_file, storage, start, stop):
    for key, data in storage.items():
        if key == "size": continue
        if key in output_file:
            output_file[key][start: stop] = data
        else:
            raise ValueError(f"Dataset {key} not found in output h5 file")
