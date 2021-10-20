import os, time
import h5py
import numpy as np

from SiPMStudio.processing.process_data import _chunk_range
from SiPMStudio.processing.reprocess_data import data_chunk, _process_chunk
from SiPMStudio.utils.gen_utils import tqdm_range

def output_chunk(output, h5_file, begin, end):
    data_len = h5_file["/raw/timetag"].shape[0]
    for key, value in output.items():
        h5_file[key][begin:end] = value


def reprocess_data(rank, chunk_idx, processor, h5_input, bias=None, overwrite=False, verbose=False, chunk=2000, write_size=1):
    
    start = time.time()
    num_rows = h5_input["/raw/timetag"].shape[0]
    
    for i in tqdm_range(chunk_idx[0], chunk_idx[1], position=rank, verbose=verbose):
        begin, end = _chunk_range(i, chunk, num_rows)
        storage = data_chunk(h5_input, begin, end)
        output_storage = _process_chunk(storage, processor)
        output_chunk(output_storage, h5_input, begin, end)
        processor.reset_outputs()
