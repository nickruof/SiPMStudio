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
    measurement.digitizer = digitizer



