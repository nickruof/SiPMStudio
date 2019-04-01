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

def ProcessData(data_files, 
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

    #Declare an output file and overwriting options here

    processor.digitizer = digitizer

    for i, file in enumerate(data_files):
        digitizer.load_data(df_data=file)
        chunk_idx = get_chunks(digitizer=digitizer, chunksize=chunk)
        if multiprocess:
            wait = animation.Wait("spinner")
            with ThreadPool(NCPU) as p:
                wait.start()
                p.map(partial(process_chunk, digitizer=digitizer, processor=processor), chunk_idx)
                wait.stop()
        else:
            for idx in tqdm.tqdm(chunk_idx, total=len(chunk_idx)):
                process_chunk(digitizer=digitizer, processor=processor, rows=idx)

        write_output(data_file=file, output_frame=output_df, output_dir=output_dir)

    #if multiprocess:
    #    with ThreadPool(NCPU) as p:
    #        data_list = []
    #        for i, file in enumerate(data_files):
    #            data_list.append([file, i])
    #        result = p.map(
    #            partial(process_files, digitizer=digitizer, processor=processor, chunk=chunk), data_list)
            #result_list = list(result)

    #else:
    #    for i, file in enumerate(data_files):
    #        output_df = process_files(file=[file, i], digitizer=digitizer, processor=processor, chunk=chunk)
    #        write_output(data_file=file, output_frame=output_df, output_dir=output_dir, chunk=chunk)

    elapsed = time.time() - start
    output_time(elapsed)

def get_chunks(digitizer, chunksize):
    num_chunks = 1
    if chunksize is not None:
        num_rows = sum(1 for line in open(file))
        num_chunks = math.ceil(num_rows / chunksize)
    else:
        chunk = sum(1 for line in open(file))
    df_size = os.path.getsize(file)

    row_list = []
    for i in range(num_chunks):
        if (i+1)*chunksize < digitizer.df_data.shape[0]:
            row_list.append([i*chunksize, (i+1)*chunksize])
        else:
            row_list.append([i*chunksize])
    return row_list

def process_files(file, digitizer, processor, chunk):
    print("Processing: " + str(file[0]))
    num_chunks = 1
    if chunk is not None:
        num_rows = sum(1 for line in open(file[0]))
        num_chunks = math.ceil(num_rows / chunk)
    else:
        chunk = sum(1 for line in open(file[0]))
    digitizer.load_data(df_data=file[0], chunksize=chunk)
    df_size = os.path.getsize(file[0])

    
    for i in range(num_chunks):
        if (i+1)*chunk < digitizer.df_data.shape[0]:
            processor.set_processor(digitizer=digitizer, rows=[i*chunk, (i+1)*chunk])
            processor.process()
        else:
            processor.set_processor(digitizer=digitizer, rows=[i*chunk])
            processor.process()


    #for block in tqdm.tqdm(digitizer.df_data, total=num_chunks, position=int(file[1])):
        #new_chunk = process_chunk(df_data=block, processor=processor)
        #output_df = pd.concat([output_df, new_chunk], ignore_index=True)

    return digitizer.df_data

def process_chunk(digitizer, processor, rows=None):
    processor.set_processor(digitizer, rows=rows)
    processor.process()

def write_output(data_file, output_frame, output_dir):
    print("")
    print("Writing Output file ...")
    indices = [i for i, item in enumerate(data_file) if item == "/"]
    file_name = data_file[indices[-1]+1:]
    output_frame.to_csv(output_dir+"t1_"+file_name)

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
    print(" ")
    print("Time elapsed: "+str(hours)+"h "+str(minutes)+"m "+str(seconds)+"s ")



    

    
