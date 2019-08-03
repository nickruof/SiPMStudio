import os, time
import math
import tqdm
import pandas as pd


def process_bigdata(path, data_files, processor, digitizer, overwrite=False, output_dir=None, verbose=False, chunk=2000):

    print("Starting SiPMStudio processing ... ")
    print("Input Path: ", path)
    output_dir = os.getcwd() if output_dir is None else output_dir
    print("Output Path: ", output_dir)
    print("Input Files: ", data_files)
    file_sizes = []
    for file_name in data_files:
        memory_size = os.path.getsize(path+"/"+file_name)
        file_sizes.append(str(memory_size/1e6)+" MB")
    print("File Sizes: ", file_sizes)
    if overwrite is True:
        for file_name in data_files:
            if os.path.isfile(path+"/"+file_name):
                os.remove(path+"/"+file_name)

    digitizer_type = digitizer.__class__
    processor.digitizer = digitizer_type()

    start = time.time()
    # -----Processing Begins Here!---------------------------------

    for file in data_files:
        destination = path+"/"+file
        print("Processing: "+file)
        store = pd.HDFStore(destination)
        num_rows = store.get_storer("dataset").shape[0]
        for i in tqdm.tqdm(range(num_rows//chunk + 1)):
            df_chunk = store.select("dataset", start=i*chunk, stop=(i+1)*chunk)
            processor.digitizer.load_data(df_chunk)
            output_df = _process_chunk(processor=processor)
            _output_to_file(data_file=destination, output_frame=output_df, output_dir=output_dir, write_allocs=i)
        processor.digitizer.clear_data()
        digitizer.clear_data()
        store.close()

    print("Processing Finished! ...")
    _output_time(time.time() - start)


def _get_num_chunks(digitizer, chunksize):
    num_chunks = 1
    if chunksize is not None:
        num_rows = sum(1 for row in digitizer.df_data.to_numpy())
        num_chunks = math.ceil(num_rows / chunksize)
    else:
        num_chunks = sum(1 for row in digitizer.df_data.to_numpy())

    return num_chunks


def _process_chunk(processor, rows=None):
    processor.set_processor(rows=rows)
    return processor.process()


def _output_to_file(data_file, output_frame, output_dir, write_allocs=0, prefix="t2"):
    indices = [i for i, item in enumerate(data_file) if item == "/"]
    file_name = data_file[indices[-1]+1:]
    output_frame.columns = output_frame.columns.astype(str)
    new_file_name = ""
    if (prefix in file_name) & (file_name.endswith(".h5")):
        new_file_name = file_name
    elif ("t1" in file_name) & (file_name.endswith(".h5")):
        new_file_name = file_name.replace("t1", prefix)
    else:
        new_file_name = prefix+"_"+file_name[:-4]+".h5"

    minimum_size = {"index": 10}
    with pd.HDFStore(output_dir+"/"+new_file_name) as store:
        if write_allocs == 0:
            store.put(key="dataset", value=output_frame, format="table", min_itemsize=minimum_size)
        else:
            store.put(key="dataset", value=output_frame, format="table", append=True, min_itemsize=minimum_size)


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
