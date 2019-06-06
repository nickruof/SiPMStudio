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

def Experiment(files,
                measurement_arrays, # replace with measurement settings
                utility_belt,
                digitizer,
                overwrite=True,
                verbose=False):

    print("Running Experiment ...")
    print("Files: ", files)

    start = time.time()

    for i, file in enumerate(tqdm.tqdm(files, total=len(files))):
        digitizer.load_data(df_data=file)
        measurement_arrays[i].set_array(digitizer=digitizer)
        _process(file=file, digitizer=digitizer, measurement_array=measurement_arrays[i], utility_belt=utility_belt)

    _output_time(time.time()-start)


def _process(file, digitizer, measurement_array, utility_belt):
    measurement_array.file = file
    measurement_array.set_array(digitizer=digitizer)
    measurement_array.run(utility_belt)
    utility_belt.clear()

