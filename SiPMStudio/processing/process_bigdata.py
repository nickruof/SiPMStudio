import os, time
import math
import copy
import tqdm
import animation
import pandas as pd
import multiprocessing as mp

from functools import partial
from pathos.threading import ThreadPool


def process_data(path,
                data_files,
                processor,
                digitizer,
                output_dir=None,
                overwrite=True,
                verbose=False,
                chunk=2000):

    print("Starting SiPMStudio processing ... ")
    print("Input Path: ", path)
    print("Input Files: ", data_files)

    NCPU = mp.cpu_count()

    start = time.time()
    output_dir = os.getcwd() if output_dir is None else output_dir

    # Declare an output file and overwriting options here

    processor.digitizer = copy.deepcopy(digitizer)
    processor.file = path+"/"+"settings.json"
    data_chunks = []

    for file in data_files:
        destination = path+"/"+file
        print("Loading: "+file)
        digitizer.load_data(df_data=destination, chunksize=chunk)
        for i, df_chunk in enumerate(processor.digitizer.df_data):
            processor.digitizer.load_data(df_chunk)
            output_df = _process_chunk(processor=processor, rows=None)
            _output_to_file(data_file=destination, output_frame=output_df, output_dir=output_dir, write_allocs=i)
        processor.digitizer.clear_data()

    _output_time(time.time() - start)


def _get_chunks(digitizer, chunksize):
    num_chunks = 1
    if chunksize is not None:
        num_rows = sum(1 for row in digitizer.df_data.to_numpy())
        num_chunks = math.ceil(num_rows / chunksize)
    else:
        num_chunks = sum(1 for row in digitizer.df_data.to_numpy())

    row_list = []
    for i in range(num_chunks):
        if (i+1)*chunksize < digitizer.df_data.shape[0]:
            row_list.append([i*chunksize, (i+1)*chunksize])
        else:
            row_list.append([i*chunksize])
    return row_list


def _process_chunk(rows, processor):
    processor.set_processor(rows=rows)
    return processor.process()


def _output_to_file(data_file, output_frame, output_dir, write_allocs=0):
    indices = [i for i, item in enumerate(data_file) if item == "/"]
    file_name = data_file[indices[-1]+1:]
    output_frame.columns = output_frame.columns.astype(str)
    new_file_name = ""
    if ("t2" in file_name) & (file_name.endswith(".h5")):
        new_file_name = file_name
    elif ("t1" in file_name) & (file_name.endswith(".h5")):
        new_file_name = file_name.replace("t1", "t2")
    else:
        new_file_name = "t2_"+file_name[:-4]+".h5"

    if write_allocs == 0:
        output_frame.to_hdf(path_or_buf=output_dir+"/"+new_file_name, key="dataset", mode="w", table=True)
    else:
        store = pd.HDFStore(output_dir+"/"+new_file_name)
        store.append(key="dataset", value=output_frame, format="table", data_columns=True)


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
