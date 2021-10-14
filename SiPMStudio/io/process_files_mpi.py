import os, time
import json
import argparse
import h5py

from mpi4py import MPI

from SiPMStudio.processing.processor import Processor, load_functions
from SiPMStudio.processing.process_data import process_data

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def process_files_mpi(settings, proc_file, bias=None, overwrite=True, chunk=4000, write_size=2, verbose=True):

    path = settings["output_path_raw"]
    path_t2 = settings["output_path_t2"]
    data_files = []
    output_files = []

    base_name = settings["file_base_name"]
    for entry in settings["init_info"]:
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
