import os, re, sys, time
import numpy as np
import pandas as pd
import multiprocessing as mp

from SiPMStudio.core import data_loading
from SiPMStudio.core import digitizers
from SiPMStudio.core import devices

def ProcessData(data_file, 
                processor, 
                digitizers=None, 
                output_dir=None, 
                overwrite=True, 
                verbose=False,
                multiprocess=True,
                chunk = 3000):
    
