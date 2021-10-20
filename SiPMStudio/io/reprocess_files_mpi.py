import os
import glob
import json
import argparse
import h5py

from mpi4py import MPI

from SiPMStudio.processing.processor import Processor
from SiPMStudio.io.reprocess_files_mpi import load_functions
from SiPMStudio.processing.reprocess_data_mpi import reprocess_data

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def _chunk_indices(rank, size, chunk, num_rows):
    total_indices = num_rows // chunk + 1
    idx_per_rank = total_indices // size
    chunk_dict = {}
    for idx in range(0, size):
        chunk_dict[idx] = [idx * idx_per_rank, idx * idx_per_rank + idx_per_rank - 1]
        if idx == (size - 1):
            chunk_dict[idx][1] += 1

    while chunk_dict[size-1][1] < total_indices:
        for idx in range(0, size):
            if idx == 0:
                chunk_dict[idx][1] += 1
            else:
                chunk_dict[idx][0] += 1
                chunk_dict[idx][1] += 1
    
    return chunk_dict[rank][0], chunk_dict[rank][1]


def _init_new_output(h5_file, proc_dict, num_rows, waveform_size):
    for i, output in enumerate(proc_dict["save_output"]):
        if proc_dict["output_shape"][i] == 2:
            h5_file.create_dataset(output, (num_rows, waveform_size))
        elif proc_dict["output_shape"][i] == 1:
            h5_file.create_dataset(output, (num_rows,))
        else:
            raise ValueError("Output shape must be 1 or 2")


def reprocess_mpi(settings_dict, proc_dict, pattern=None, file_name=None, chunk=4000, write_size=2, verbose=True):
    file_path = settings_dict["output_path_t2"]
    file_list = None
    processor = Processor()
    if file_name is None:
        file_list = glob.glob(f"{file_path}/*.h5")
    else:
        file_list = os.path.join(file_path, file_name)

    for output in proc_dict["save_output"]:
        processor.add_to_file(output)
    
    for i, file_name in enumerate(file_list):
        head_dir, tail_name = os.path.split(file_name)
        if pattern is None:
            load_functions(tail_name, proc_dict, processor)
        elif any(bias in tail_name for bias in pattern):
            load_functions(tail_name, proc_dict, processor)
        else:
            continue
    
        h5_file = h5py.File(file_name, "r+", driver="mpio", comm=comm)
        num_rows = h5_file["/raw/timetag"][:].shape[0]
        waveform_size = h5_file["/processed/blr_wf"].shape[1]
        _init_new_output(h5_file, proc_dict, num_rows, waveform_size)
        [begin, end] = _chunk_indices(rank, size, chunk, num_rows)

        reprocess_data(rank, [begin, end], processor, h5_file, verbose, chunk, write_size)
        processor.clear()
        h5_file.close()


def reprocess_files(settings_file, proc_file, verbose=False, pattern=None):

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    proc_dict = None
    with open(proc_file, "r") as json_file:
        proc_dict = json.load(json_file)

    reprocess_mpi(settings_dict, proc_dict, verbose=verbose, pattern=pattern)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--procs", help="processor settings file name")
    parser.add_argument("--verbose", help="set verbosity to True or False", type=bool)
    parser.add_argument("--bias", help="bias name in file pattern")
    args = parser.parse_args()

    bias_list = None
    if args.bias is not None:
        bias_list = [str(i) for i in args.bias.split(",")]

    reprocess_files(args.settings, args.procs, verbose=args.verbose, pattern=bias_list)