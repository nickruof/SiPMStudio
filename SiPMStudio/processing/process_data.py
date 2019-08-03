import os, time
import tqdm
import pandas as pd


def process_data(path, data_files, processor, digitizer, overwrite=False, output_dir=None, verbose=False, chunk=2000, write_size=1):

    print("Starting SiPMStudio processing ... ")
    print("Input Path: ", path)
    output_dir = os.getcwd() if output_dir is None else output_dir
    print("Output Path: ", output_dir)
    print("Input Files: ", data_files)
    file_sizes = []
    for file_name in data_files:
        memory_size = os.path.getsize(path+"/"+file_name)
        memory_size = round(memory_size/1e6)
        file_sizes.append(str(memory_size)+" MB")
    print("File Sizes: ", file_sizes)
    if overwrite is True:
        for file_name in _list_files(output_dir):
            os.remove(output_dir+"/"+file_name)

    digitizer_type = digitizer.__class__
    processor.digitizer = digitizer_type()

    start = time.time()
    # -----Processing Begins Here!---------------------------------

    for file in data_files:
        destination = path+"/"+file
        print("Processing: "+file)
        store = pd.HDFStore(destination)
        num_rows = store.get_storer("dataset").shape[0]
        df_storage = pd.DataFrame()
        for i in tqdm.tqdm(range(num_rows//chunk + 1)):
            df_chunk = store.select("dataset", start=i*chunk, stop=(i+1)*chunk)
            processor.digitizer.load_data(df_chunk)
            output_df = _process_chunk(processor=processor)
            _output_chunk(destination, output_df, df_storage, output_dir, [chunk, write_size])
        processor.digitizer.clear_data()
        digitizer.clear_data()
        store.close()

    print("Processing Finished! ...")
    print("Output Path: ", output_dir)
    print("Output Files: ", _list_files(output_dir))
    _output_time(time.time() - start)


def _process_chunk(processor, rows=None):
    processor.set_processor(rows=rows)
    return processor.process()


def _output_chunk(data_file, chunk_frame, output_frame, output_dir, read_write, prefix="t2"):
    if read_write[1] == 1:
        _output_to_file(data_file, chunk_frame, output_dir, prefix)
    else:
        if output_frame.shape[0] >= read_write[0]*read_write[1]:
            _output_to_file(data_file, output_frame, output_dir, prefix)
            output_frame = output_frame.iloc[0:0]
        else:
            output_frame.append(chunk_frame)


def _output_to_file(data_file, output_frame, output_dir, prefix="t2"):
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
        if "dataset" in store:
            store.put(key="dataset", value=output_frame, format="table", append=True, min_itemsize=minimum_size)
        else:
            store.put(key="dataset", value=output_frame, format="table", append=False, min_itemsize=minimum_size)


def _list_files(path):
    all_files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    h5_files = [file for file in all_files if file.endswith(".h5")]
    return h5_files


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