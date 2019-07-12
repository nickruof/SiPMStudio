import tqdm
import pandas as pd


def process_metadata(path, metadata_files, digitizer, output_dir=None, verbose=False):

    destinations = _attach_path(path, metadata_files)

    for file_name in tqdm.tqdm(destinations, total=len(destinations)):
        dataframe_list = []
        event_size = digitizer.get_event_size(file_name)

        with open(file_name, "rb") as metadata_file:
            event_data_bytes = metadata_file.read(event_size)
            while event_data_bytes != b"":
                dataframe_list.append(digitizer.get_event(event_data_bytes))
        output_dataframe = pd.concat(dataframe_list)
        _output_to_file(output_dir, output_dataframe)


def _attach_path(path, files):
    new_files = []
    for file in files:
        new_files.append(path+"/"+file)
    return new_files


def _output_to_file(output_path, input_dataframe):
    indices = [i for i, item in enumerate(data_file) if item == "/"]
    file_name = data_file[indices[-1] + 1:]
    output_frame.columns = output_frame.columns.astype(str)
    output_frame.to_hdf(path_or_buf=output_path + "t1_" + file_name[:-4] + ".h5", key="dataset", mode="w", table=True)


