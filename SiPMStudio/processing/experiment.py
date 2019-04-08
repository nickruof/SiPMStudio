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

from functools import partial
from pathos.threading import ThreadPool

def Experiment(param_files,
                wave_files, 
                measurement_array,
                utility_belt, 
                digitizer,
                overwrite=True,
                multiprocess=True, 
                verbose=False,
                chunk=2000):

    print("Running Experiment ...")
    print("Param files: ", param_files)
    print("Wave files: ", wave_files)

    start = time.time()
    measurement.params_digitizer = digitizer
    measurement.waves_digitizer = digitizer

    if param_files is None:
        param_files = [None]*len(wave_files)
    elif wave_files is None:
        wave_files = [None]*len(param_files)
    else:
        pass

    for params, waves in zip(param_files, wave_files):





    _output_time(time.time()-start)