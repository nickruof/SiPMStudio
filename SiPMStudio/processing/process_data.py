import os, time
import tqdm
import h5py
import numpy as np


def process_data(settings, processor, bias=None, overwrite=False, verbose=False, chunk=2000, write_size=1):

    path = settings["output_path_t1"]
    path_t2 = settings["output_path_t2"]
    data_files = []
    output_files = []

    base_name = settings["file_base_name"]
    for entry in settings["init_info"]:
        bias_label = entry["bias"]
        if bias is None:
            data_files.append(f"t1_{base_name}_{bias_label}.h5")
            output_files.append(f"t2_{base_name}_{bias_label}")
        elif entry["bias"] in bias:
            data_files.append(f"t1_{base_name}_{bias_label}.h5")
            output_files.append(f"t2_{base_name}_{bias_label}.h5")
        else:
            pass

    if verbose:
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

    for file in data_files:
        destination = os.path.join(path, file)
        output_destination = os.path.join(path_t2, output_files[i])
        if verbose:
            print(f"Processing: {file}")
        h5_file = h5py.File(destination, "r")
        num_rows = h5_file["/raw/waveforms"][:].shape[0]
        df_storage = {}
        for i in tqdm.tqdm(range(num_rows//chunk + 1)):
            begin, end = _chunk_range(i, chunk, num_rows)
            wf_chunk = h5_file["/raw/waveforms"][begin:end]
            time_chunk = h5_file["/raw/timetag"][begin:end]
            output_data = _process_chunk(wf_chunk, time_chunk, processor=processor)
            _output_chunk(h5_file, output_destination, output_data, df_storage, write_size, num_rows, chunk, end)
            processor.reset_output()
        _copy_to_t2(["bias", "/raw/timetag"], ["bias", "/processed/timetag"], h5_file, output_destination)
        h5_file.close()

    if verbose:
        print("Processing Finished! ...")
        print("Output Files: ", [file.replace("t1", "t2") for file in data_files])
        _output_time(time.time() - start)


def _chunk_range(index, chunk, num_rows):
    start = index * chunk
    stop = (index+1) * chunk
    if stop >= num_rows - 1:
        stop = num_rows - 1
    return start, stop


def _process_chunk(wf_chunk, time_chunk, processor, rows=None):
    processor.init_outputs({"/processed/waveforms": wf_chunk, "/raw/timetag": time_chunk})
    processor.set_processor(wf_chunk, rows=rows)
    return processor.process()


def _output_chunk(data_file, output_file, chunk_data, storage, write_size, num_rows, chunk, stop):
    for output in chunk_data.keys():
        if output not in storage:
            storage[output] = []
        if (write_size == 1) | (num_rows < chunk):
            _output_to_file(data_file, output_file, chunk_data, output)
        else:
            if stop >= num_rows-1:
                storage[output].append(chunk_data[output])
                _output_to_file(data_file, output_file, storage, output)
                storage.clear()
            else:
                storage[output].append(chunk_data[output])
                if len(storage) == write_size:
                    _output_to_file(data_file, output_file, storage, output)
                    storage.clear()


def _copy_to_t2(raw_names, process_names, h5_file, output_destination):
    if len(raw_names) != len(process_names):
        raise AttributeError("raw names and process names not the same length")
    with h5py.File(output_destination, "a") as output_file:
        for i, name in enumerate(raw_names):
            output_file.create_dataset(process_names[i], data=h5_file[name])


def _output_to_file(data_file, output_filename, storage, output_name):
    output_data = None
    if isinstance(storage[output_name], list):
        output_data = np.concatenate(storage[output_name])
    else:
        output_data = storage[output_name]
    if output_name in data_file:
        with h5py.File(output_filename, "a") as output_file:
            output_file[output_name].resize(output_file[output_name].shape[0]+output_data.shape[0], axis=0)
            output_file[output_name][-output_data.shape[0]:] = output_data
    else:
        with h5py.File(output_filename, "w") as output_file:
            output_file.create_dataset(output_name, data=output_data, maxshape=(None, None))


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
    print(f"Time elapsed {hours}h {minutes}m {seconds}s")
    print(" ")