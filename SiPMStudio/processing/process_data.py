import os, time
import h5py
import numpy as np

from SiPMStudio.utils.gen_utils import tqdm_range

def process_data(settings, processor, bias=None, overwrite=False, verbose=False, chunk=2000, write_size=1):

    path = settings["output_path_raw"]
    path_t2 = settings["output_path_t2"]
    data_files = []
    output_files = []

    base_name = settings["file_base_name"]
    for entry in settings["init_info"]:
        bias_label = entry["bias"]
        if bias is None:
            data_files.append(f"raw_{base_name}_{bias_label}.h5")
            output_files.append(f"t2_{base_name}_{bias_label}.h5")
        elif entry["bias"] in bias:
            data_files.append(f"raw_{base_name}_{bias_label}.h5")
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

    for idx, file in enumerate(data_files):
        destination = os.path.join(path, file)
        output_destination = os.path.join(path_t2, output_files[idx])
        if verbose:
            print(f"Processing: {file}")
        h5_file = h5py.File(destination, "r")
        h5_output_file = h5py.File(output_destination, "a")
        num_rows = h5_file["n_events"][()]
        data_storage = {"size": 0}
        for i in tqdm_range(0, num_rows//chunk + 1, verbose=verbose):
            begin, end = _chunk_range(i, chunk, num_rows)
            _initialize_outputs(idx, settings, h5_file, processor, begin, end)
            output_data = processor.process()
            _output_chunk(h5_output_file, output_data, data_storage, write_size, num_rows, chunk, end)
            processor.reset_outputs()
        _copy_to_t2(h5_file, h5_output_file)
        _output_date(output_destination, "process_date")
        h5_file.close()
        h5_output_file.close()

    if verbose:
        print("Processing Finished! ...")
        print("Output Files: ", [file.replace("raw", "t2") for file in data_files])
        _output_time(time.time() - start)


def _chunk_range(index, chunk, num_rows):
    start = index * chunk
    stop = (index+1) * chunk
    if stop >= num_rows:
        stop = num_rows
    return start, stop


def _initialize_outputs(idx, settings, h5_file, processor, begin, end):
    data_dict = {}
    for channel in settings["init_info"][idx]["channels"]:
        data_dict["timetag"] = h5_file["timetag"][begin: end]
        data_dict[f"/raw/{channel}/waveforms"] = h5_file[f"/raw/{channel}/waveforms"][begin: end]
    processor.init_outputs(data_dict)


def _output_chunk(output_file, chunk_data, storage, write_size, num_rows, chunk, stop):
    output_to_file = False
    if (write_size == 1) | (num_rows < chunk):
        output_to_file = True
    elif stop >= num_rows-1:
        output_to_file = True
    elif storage["size"] == (write_size - 1):
        output_to_file = True

    for i, output in enumerate(chunk_data.keys()):
        if output not in storage:
            storage[output] = []
        storage[output].append(chunk_data[output])
        if i == 0:
            storage["size"] = len(storage[output])
        if output_to_file:
            storage[output] = np.concatenate(storage[output])
    if output_to_file:
        _output_to_file(output_file, storage)
        storage.clear()
        storage["size"] = 0


def _copy_to_t2(h5_file, output_file):
    for key in h5_file.keys():
        if key != "raw":
            output_file.create_dataset(key, data=h5_file[key])


def _output_to_file(output_file, storage):
    for key, data in storage.items():
        if key == "size": continue
        if key in output_file:
            output_file[key].resize(output_file[key].shape[0]+data.shape[0], axis=0)
            output_file[key][-data.shape[0]:] = data
        else:
            if len(data.shape) == 2:
                output_file.create_dataset(key, data=data, maxshape=(None, None))
            elif len(data.shape) == 1:
                output_file.create_dataset(key, data=data, maxshape = (None,))
            else:
                raise ValueError(f"Dimension of output data {data.shape} must be 1 or 2")


def _output_date(output_destination, label=None):
    with h5py.File(output_destination, "a") as output_file:
        if label is None:
            label = "date"
        if label not in output_file.keys():
            output_file.create_dataset(label, data=int(time.time()))
        else:
            output_file[label] = int(time.time())


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
