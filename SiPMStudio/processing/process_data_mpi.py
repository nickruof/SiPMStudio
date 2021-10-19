import os, time
import h5py
import numpy as np

from SiPMStudio.utils.gen_utils import tqdm_range

def process_data(rank, chunk_idx, processor, h5_input, h5_output, bias=None, overwrite=False, verbose=False, chunk=2000, write_size=1):

    start = time.time()
    # -----Processing Begins Here!---------------------------------
    num_rows = h5_input["/raw/timetag"][:].shape[0]
    data_storage = {"size": 0}
    write_count = 0
    write_begin = 0
    write_end = 0
    for i in tqdm_range(chunk_idx[0], chunk_idx[1], position=rank, verbose=verbose):
        begin, end = _chunk_range(i, chunk, num_rows)
        if write_count == 0:
            write_count += 1
            write_begin = begin
            write_end = end
        
        if (write_count == write_size) & (write_size > 1):
            write_count = 0
        elif write_size == 1:
            write_count = 0
        else:
            write_count += 1
            write_end += chunk
        wf_chunk = h5_input["/raw/waveforms"][begin:end]
        time_chunk = h5_input["/raw/timetag"][begin:end]
        output_data = _process_chunk(wf_chunk, time_chunk, processor=processor)
        _output_chunk(h5_output, output_data, data_storage, write_size, num_rows, chunk, write_begin, write_end)
        processor.reset_outputs()


def _chunk_range(index, chunk, num_rows):
    start = index * chunk
    stop = (index+1) * chunk
    if stop >= num_rows:
        stop = num_rows
    return start, stop


def _process_chunk(wf_chunk, time_chunk, processor, rows=None):
    processor.init_outputs({"/raw/waveforms": wf_chunk, "/raw/timetag": time_chunk})
    return processor.process()


def _output_chunk(output_file, chunk_data, storage, write_size, num_rows, chunk, start, stop):
    output_to_file = False
    if (write_size == 1) | (num_rows < chunk):
        output_to_file = True
    elif stop >= num_rows-1:
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
