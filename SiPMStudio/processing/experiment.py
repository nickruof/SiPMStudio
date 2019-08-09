import os, re, sys, time
import math
import tqdm
import animation
import numpy as np
import pandas as pd
import multiprocessing as mp

from SiPMStudio.core import data_loading
from SiPMStudio.core import digitizers
from SiPMStudio.core import devices
from SiPMStudio.processing import measurement
from SiPMStudio.processing.process_data import _output_time


def experiment(path, files, settings_files, measurement_array, digitizer, output_dir=None, overwrite=True, verbose=False):

    print("Running Experiment ...")
    print("Input Path: ", path)
    print("Files: ", files)
    output_dir = os.getcwd() if output_dir is None else output_dir
    print("Output Path: ", output_dir)
    print("Input Files: ", files)
    file_sizes = []
    for file_name in files:
        memory_size = os.path.getsize(path+"/"+file_name)
        memory_size = round(memory_size/1e6)
        file_sizes.append(str(memory_size)+" MB")
    print("File Sizes: ", file_sizes)

    start = time.time()

    for i, file in enumerate(tqdm.tqdm(files, total=len(files))):
        destination = path+"/"+file
        digitizer.load_data(df_data=destination)
        measurement_array.set_array(digitizer=digitizer)
        _process(path+"/"+settings_files[i], digitizer, measurement_array)

    _output_time(time.time()-start)


def _process(settings_file, digitizer, measurement_array):
    measurement_array.file = settings_file
    measurement_array.set_array(digitizer=digitizer)
    measurement_array.run()

