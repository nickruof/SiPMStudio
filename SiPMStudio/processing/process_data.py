import os, re, sys, time
import math
import tqdm
import numpy as np
import pandas as pd

from SiPMStudio.core import data_loading
from SiPMStudio.core import digitizers
from SiPMStudio.core import devices

from functools import partial

def ProcessData(data_files, 
                processor, 
                digitizer, 
                output_dir=None, 
                overwrite=True, 
                verbose=False,
                chunk=2000):

    print("Starting SiPMStudio processing ... ")
    print("Input files: ", data_files)

    start = time.time()
    output_dir = os.getcwd() if output_dir is None else output_dir

    #Declare an output file and overwriting options here

    processor.digitizer = digitizer

    for file in data_files:
        num_chunks = 1
        if chunk is not None:
            num_rows = sum(1 for line in open(file))
            num_chunks = math.ceil(num_rows / chunk)
        else:
            chunk = sum(1 for line in open(file))
        digitizer.load_data(df_data=file, chunksize=chunk)
        df_size = os.path.getsize(file)
        output_df = pd.DataFrame()

        for block in tqdm.tqdm(digitizer.df_data, total=num_chunks):
            new_chunk = process_chunk(df_data=block, processor=processor)
            output_df = pd.concat([output_df, new_chunk], ignore_index=True)

        write_output(data_file=file, output_frame=output_df, output_dir=output_dir, chunk=chunk)

    elapsed = time.time() - start
    output_time(elapsed)


def process_chunk(df_data, processor):
    df_data = df_data.drop([3], axis=1)
    df_data = df_data.reindex(axis=1)
    [processor.calcs, processor.waves] = np.split(df_data, [3], axis=1)
    processor.process()
    return pd.concat([processor.calcs, processor.waves], axis=1)

def write_output(data_file, output_frame, output_dir, chunk):
    indices = [i for i, item in enumerate(data_file) if item == "/"]
    file_name = data_file[indices[-1]+1:]
    output_frame.to_csv(output_dir+"t1_"+file_name, chunksize=chunk)

def output_time(delta_seconds):
    temp_seconds = delta_seconds
    hours = 0
    minutes = 0
    seconds = 0

    while temp_seconds >= 3600:
        temp_seconds = temp_seconds - 3600
        hours = hours + 1

    while temp_seconds >= 60:
        temp_seconds = temp_seconds - 60
        minutes = minutes + 1
    seconds = round(temp_seconds, 1)
    print("Time elapsed: "+str(hours)+"h "+str(minutes)+"m "+str(seconds)+"s ")



    

    
