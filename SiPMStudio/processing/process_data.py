import os, re, sys, time
import tqdm
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
        output_df = pd.DataFrame()
        for block in tqdm(d.df_data):
            if multiprocess:
                with mp.Pool(NCPU) as p:
                    new_chunk = p.map(partial(process_chunk, block, processor))
                output_df = pd.concat([output_df, new_chunk])
            else:
                new_chunk = process_chunk(block, processor)
                output_df = pd.concat([output_df, new_chunk])

    elapsed = time.time() - start
    print("Time elapsed: "+str(elapsed))

    output_df.to_csv(output_dir+"t1_"+data_file)

    return output_df


def process_chunk(df_data, processor):
    df_data = df_data.drop(["FLAGS"], axis=1)
    [processor.calcs, processor.waves] = np.split(blocks, [3])
    processor.process()
    return pd.concat([processor.calcs, processor.waves], axis=1)

    

    
