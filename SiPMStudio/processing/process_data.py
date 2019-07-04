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


def process_data(data_files,
                processor, 
                digitizer, 
                output_dir=None, 
                overwrite=True,
                multiprocess=True, 
                verbose=False,
                chunk=2000):

    print("Starting SiPMStudio processing ... ")
    print("Input files: ", data_files)

    NCPU = mp.cpu_count()

    start = time.time()
    output_dir = os.getcwd() if output_dir is None else output_dir

    # Declare an output file and overwriting options here

    processor.digitizer = digitizer

    for file in data_files:
        digitizer.load_data(df_data=file)
        chunk_idx = _get_chunks(file=file, digitizer=digitizer, chunksize=chunk)
        if multiprocess:
            wait = animation.Wait(animation="elipses", text="Multiprocessing")
            with ThreadPool(NCPU) as p:
                wait.start()
                p.map(partial(_process_chunk, digitizer=digitizer, processor=processor,
                              out_frame=data_chunks), chunk_idx)
                wait.stop()
            _output_time(time.time()-start)
        else:
            for idx in tqdm.tqdm(chunk_idx, total=len(chunk_idx)):
                _process_chunk(file=file, processor=processor, rows=idx)

        print("Assembling Output Dataframe!")
        # _write_output(data_file=file, output_frame=output, output_dir=output_dir)

    _output_time(time.time() - start)


def _get_chunks(file, digitizer, chunksize):
    num_chunks = 1
    if chunksize is not None:
        num_rows = sum(1 for line in open(file))
        num_chunks = math.ceil(num_rows / chunksize)
    else:
        num_chunks = sum(1 for line in open(file))

    row_list = []
    for i in range(num_chunks):
        if (i+1)*chunksize < digitizer.df_data.shape[0]:
            row_list.append([i*chunksize, (i+1)*chunksize])
        else:
            row_list.append([i*chunksize])
    return row_list


def _process_chunk(file, rows, processor, out_frame):
    processor.file = file
    processor.set_processor(rows=rows)
    processor.digitizer.update(params=processor.params, waves=processor.waves)


def _write_output(data_file, output_frame, output_dir):
    print("")
    print("Writing Output file ...")
    indices = [i for i, item in enumerate(data_file) if item == "/"]
    file_name = data_file[indices[-1]+1:]
    output_frame.columns = output_frame.columns.astype(str)
    output_frame.to_hdf(path_or_buf=output_dir+"t1_"+file_name[:-4]+".h5", key="dataset", mode="w", table=True)


def _output_time(delta_seconds):
    temp_seconds = delta_seconds
    hours = 0
    minutes = 0

    while temp_seconds >= 3600:
        temp_seconds = temp_seconds - 3600
        hours = hours + 1

    while temp_seconds >= 60:
        temp_seconds = temp_seconds - 60
        minutes = minutes + 1
    seconds = round(temp_seconds, 1)
    print(" ")
    print("Time elapsed: "+str(hours)+"h "+str(minutes)+"m "+str(seconds)+"s ")
