import os, re, sys, time
import math
import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp

from SiPMStudio.core import data_loading
from SiPMStudio.core import digitizers
from SiPMStudio.core import devices

from functools import partial

def ProcessData(data_file, 
                processor, 
                digitizers=None, 
                output_dir=None, 
                overwrite=True, 
                verbose=False,
                multiprocess=True,
                chunk=3000):

    print("Starting SiPMStudio processing ... ")
    print("Input file: "+data_file)

    start = time.time()
    in_dir = os.path.dirname(data_file)
    output_dir = os.getcwd() if output_dir is None else output_dir

    CHUNKSIZE = chunk
    NCPU = mp.cpu_count()

    #Declare an output file and overwriting options here

    for d in digitizers:
        processor.digitizer = d
        d.load_data(df_data=data_file, chunksize=chunk)
        df_size = os.path.getsize(data_file)
        num_rows = sum(1 for line in open(data_file))
        num_chunks = math.ceil(num_rows / chunk)

        output_df = pd.DataFrame()
        for block in tqdm.tqdm(d.df_data, total=num_chunks):
            print(type(block))
            if multiprocess:
                async_list = []
                with mp.Pool(NCPU) as p:
                    async_proc = p.apply_async(partial(process_chunk, processor=processor), [block])
                    async_list.append(async_proc)

                #output_df = pd.concat([output_df, new_chunk], ignore_index=True)
            else:
                new_chunk = process_chunk(df_data=block, processor=processor)
                output_df = pd.concat([output_df, new_chunk], ignore_index=True)

    elapsed = round(time.time() - start, 1)
    print("Time elapsed: "+str(elapsed)+" s")
    #output_df.to_csv(output_dir+"t1_"+data_file)

    #return output_df


def process_chunk(df_data, processor):
    df_data = df_data.drop([3], axis=1)
    df_data = df_data.reindex(axis=1)
    [processor.calcs, processor.waves] = np.split(df_data, [3], axis=1)
    processor.process()
    return pd.concat([processor.calcs, processor.waves], axis=1)

def retrieve_dataframe(asyncs):
    output_frame = pd.DataFrame()
    for proc in asyncs:
        new_chunk = proc.get()
        output_frame = pd.concat([output_frame, new_chunk], ignore_index=True)
    return output_frame

    

    
