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
                measurement_array,
                digitizer,
                overwrite=True,
                verbose=False):

    print("Running Experiment ...")
    print("Files: ", files)

    start = time.time()

    for i, file in enumerate(tqdm.tqdm(files, total=len(files))):
        digitizer.load_data(df_data=file)
        measurement_array.set_array(digitizer=digitizer)
        _process(file=file, digitizer=digitizer, measurement_array=measurement_array)

    _output_time(time.time()-start)


def _process(file, digitizer, measurement_array):
    measurement_array.file = file
    measurement_array.set_array(digitizer=digitizer)
    measurement_array.run()

