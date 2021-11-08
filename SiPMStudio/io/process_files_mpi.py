import os, time
import json
import argparse
import h5py
import numpy as np

from mpi4py import MPI

from SiPMStudio.processing.processor import Processor, load_functions
from SiPMStudio.processing.process_data import _copy_to_t2
from SiPMStudio.processing.process_data_mpi import process_data

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


def _init_output(h5_input, h5_output, proc_dict):
    n_entries = h5_input["n_events"][()]
    for channel in h5_input["/raw/channels"].keys():
        wf_length = h5_input[f"/raw/channels/{channel}/wf_len"][()]
        for output in proc_dict["save_output"]:
            h5_output.create_dataset(output, (n_entries,))
        for output in proc_dict["save_waveforms"]:
            h5_output.create_dataset(output, (n_entries, wf_length))


def process_files_mpi(settings, proc_file, bias=None, overwrite=True, chunk=4000, write_size=2, verbose=True):

    settings_dict = None
    with open(settings, "r") as json_file:
        settings_dict = json.load(json_file)

    path = settings_dict["output_path_raw"]
    path_t2 = settings_dict["output_path_t2"]
    data_files = []
    output_files = []

    base_name = settings_dict["file_base_name"]
    for entry in settings_dict["init_info"]:
        bias_label = entry["bias"]
        if bias is None:
            data_files.append(f"raw_{base_name}_{bias_label}.h5")
            output_files.append(f"t2_{base_name}_{bias_label}.h5")
        elif entry["bias"] in bias:
            data_files.append(f"raw_{base_name}_{bias_label}.h5")
            output_files.append(f"t2_{base_name}_{bias_label}.h5")
        else:
            pass

    if verbose & (rank == 0):
        print(" ")
        print("Starting SiPMStudio processing ... ")
        print("Input Path: ", path)
        print("Output Path: ", path_t2)
        print("Input Files: ", data_files)

        file_sizes = []
        for file_name in data_files:
            memory_size = os.path.getsize(path+"/"+file_name)
            memory_size = round(memory_size/1e6)
            file_sizes.append(str(memory_size)+" MB")
        print("File Sizes: ", file_sizes)

    if (overwrite is True) & (rank == 0):
        for file_name in output_files:
            destination = os.path.join(path_t2, file_name)
            if os.path.isfile(destination):
                os.remove(destination)

    processor = Processor()

    proc_dict = None
    with open(proc_file, "r") as json_file:
        proc_dict = json.load(json_file)

    load_functions(proc_dict, processor)
    for idx, file in enumerate(data_files):
        destination = os.path.join(path, file)
        output_destination = os.path.join(path_t2, output_files[idx])
        h5_file = h5py.File(destination, "r", driver="mpio", comm=comm)
        h5_output_file = h5py.File(output_destination, "a", driver="mpio", comm=comm)
        _init_output(h5_file, h5_output_file, proc_dict)
        num_rows = h5_file["n_events"][()]
        [begin, end] = _chunk_indices(rank, size, chunk, num_rows)
        process_data(comm, rank, [begin, end], processor,
                    h5_file, h5_output_file, bias,
                    overwrite, verbose, chunk, write_size)

        h5_file.close()
        h5_output_file.close()

        comm.Barrier()
        if rank == 0:
            h5_file = h5py.File(destination, "r")
            h5_output_file = h5py.File(output_destination, "a")
            _copy_to_t2(h5_file, h5_output_file)
            h5_file.close()
            h5_output_file.close()
        comm.Barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--procs", help="processor settings file name")
    parser.add_argument("--bias", help="list of biases to process, comma separated", default=None)
    parser.add_argument("--verbose", help="print extra output at runtime", type=bool, default=False)
    args = parser.parse_args()

    bias_list = None
    if args.bias is not None:
        bias_list = [int(i) for i in args.bias.split(",")]

    process_files_mpi(args.settings, args.procs, verbose=args.verbose, bias=bias_list)