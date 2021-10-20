import h5py
import numpy as np

from SiPMStudio.processing.process_data import _chunk_range
from SiPMStudio.processing.reprocess_data import data_chunk, output_chunk
from SiPMStudio.utils.gen_utils import tqdm_range


def reprocess_data(rank, chunk_idx, processor, h5_input, bias=None, overwrite=False, verbose=False, chunk=2000, write_size=1):
