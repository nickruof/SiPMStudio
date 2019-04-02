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

from functools import partial
from pathos.threading import ThreadPool

def Experiment(data_files, 
                measurement_array, 
                digitizer, 
                output_dir=None, 
                overwrite=True,
                multiprocess=True, 
                verbose=False,
                chunk=2000):