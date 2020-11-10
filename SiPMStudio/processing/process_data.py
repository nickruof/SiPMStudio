import os, time
import tqdm
import h5py
import numpy as np


def process_data(settings, processor, bias=None, overwrite=False, verbose=False, chunk=2000, write_size=1):

    path = settings["output_path_t1"]
    path_t2 = settings["output_path_t2"]
    data_files = []
    output_files = []
    for entry in settings["init_info"]:
        if bias is None:
            data_files.append("t1_" + settings["file_base_name"] + "_" + str(entry["bias"]) + ".h5")
            output_files.append("t2_" + settings["file_base_name"] + "_" + str(entry["bias"]) + ".h5")
        elif entry["bias"] in bias:
            data_files.append("t1_" + settings["file_base_name"] + "_" + str(entry["bias"]) + ".h5")
            output_files.append("t2_" + settings["file_base_name"] + "_" + str(entry["bias"]) + ".h5")
        else:
            pass

    print(" ")
    print("Starting SiPMStudio processing ... ")
    print("Input Path: ", path)
    print("Output Path: ", path_t2)
    print("Input Files: ", data_files)

    file_sizes = []
    for file_name in data_files:
        memory_size = os.path.getsize(path+"/"+file_name)
        memory_size = round(memory_size/1e6)
        file_sizes.append(str(memory_size)+" MB")
    print("File Sizes: ", file_sizes)

    if overwrite is True:
        for file_name in output_files:
            destination = os.path.join(path_t2, file_name)
            if os.path.isfile(destination):
                os.remove(destination)

    start = time.time()
    # -----Processing Begins Here!---------------------------------

    for i, file in enumerate(data_files):
        destination = os.path.join(path, file)
        output_destination = os.path.join(path_t2, output_files[i])
        print("Processing: "+file)
        h5_file = h5py.File(destination, "r")
        num_rows = h5_file["/raw/waveforms"][:].shape[0]
        df_storage = []
        for i in tqdm.tqdm(range(num_rows//chunk + 1)):
            begin, end = _chunk_range(i, chunk, num_rows)
            wf_chunk = h5_file["/raw/waveforms"][begin:end]
            output_wf = _process_chunk(wf_chunk, processor=processor)
            _output_chunk(h5_file, output_destination, output_wf, df_storage, write_size, i, num_rows, chunk, end)
        # copy_to_t2(["bias", "/raw/timetag"], ["bias", "/processed/timetag"], h5_file, output_destination)
        h5_file.close()

    print("Processing Finished! ...")
    print("Output Files: ", [file.replace("t1", "t2") for file in data_files])
    _output_time(time.time() - start)


def _chunk_range(index, chunk, num_rows):
    start = index * chunk
    stop = (index+1) * chunk
    if stop >= num_rows - 1:
        stop = num_rows - 1
    return start, stop


def _process_chunk(wf_chunk, processor, rows=None):
    processor.set_processor(wf_chunk, rows=rows)
    return processor.process()


def _output_chunk(data_file, output_file, chunk_frame, storage, write_size, iterator, num_rows, chunk, stop):
    if (write_size == 1) | (num_rows < chunk):
        _output_to_file(data_file, output_file, chunk_frame, write_size, iterator)
    else:
        if stop >= num_rows-1:
            storage.append(chunk_frame)
            _output_to_file(data_file, output_file, storage, write_size, iterator)
            storage.clear()
        else:
            storage.append(chunk_frame)
            if len(storage) == write_size:
                _output_to_file(data_file, output_file, storage, write_size, iterator)
                storage.clear()


def _copy_to_t2(raw_names, process_names, h5_file, output_destination):
    if len(raw_names) != len(process_names):
        raise AttributeError("raw names and process names not the same length")
    with h5py.File(output_destination, "w") as output_file:
        for i, name in enumerate(raw_names):
            output_file[process_names[i]] = h5_file[name]


def _output_to_file(data_file, output_filename, storage, write_size, iterator):
    output_waveforms = None
    if isinstance(storage, list):
        output_waveforms = np.concatenate(storage)
    else:
        output_waveforms = storage
    if iterator == write_size-1:
        with h5py.File(output_filename, "w") as output_file:
            output_file.create_dataset("/processed/waveforms", data=output_waveforms, maxshape=(None, None))
    else:
        with h5py.File(output_filename, "a") as output_file:
            output_file["/processed/waveforms"].resize(output_file["/processed/waveforms"].shape[0]+output_waveforms.shape[0], axis=0)
            output_file["/processed/waveforms"][-output_waveforms.shape[0]:] = output_waveforms


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
    print(" ")