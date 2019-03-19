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
                chunk=3000):

    start = time.time()
    in_dir = os.path.dirname(data_file)
    output_dir = os.getcwd() if output_dir is None else output_dir

    #Declare an output file and overwriting options here

    for d in digitizers:
        processor.digitizer = d
        d.load_data(df_data=data_file, chuncksize=chunk)
        for block in d.df_data:
            block.drop(["FLAGS"], axis=1)
            [processor.calcs, processor.waves] = np.split(block, [3])
            processor.process()
            new_chunk = pd.concat([processor.calcs, processor.waves], axis=1)
            df_result = pd.concat([df_result, new_chunk])




    

    
