import tqdm
import numpy as np


def process_metadata(metadata_files, digitizer, output_dir=None, verbose=False):

    # TODO: make more memory efficient to reduce python application memory size

    print("Processing Metadata! ...")
    print("Number of Files to Process: "+str(len(metadata_files)))
    output_dir = os.getcwd() if output_dir is None else output_dir
    print("Output Path: ", output_dir)

    for file_name in tqdm.tqdm(metadata_files, total=len(metadata_files)):
        event_rows = []
        waveform_rows = []
        event_size = digitizer.get_event_size(file_name)
        with open(file_name, "rb") as metadata_file:
            event_data_bytes = metadata_file.read(event_size)
            while event_data_bytes != b"":
                event, waveform = digitizer.get_event(event_data_bytes)
                event_rows.append(event)
                waveform_rows.append(waveform)
                event_data_bytes = metadata_file.read(event_size)
        all_data = np.concatenate((event_rows, waveform_rows), axis=1)
        output_dataframe = digitizer.create_dataframe(all_data)
        _output_to_file(file_name, output_dir, output_dataframe, digitizer)


def _output_to_file(data_file, output_path, input_dataframe, digitizer):
    indices = [i for i, item in enumerate(data_file) if item == "/"]
    file_name = data_file[indices[-1] + 1:].replace(digitizer.file_header, "")
    input_dataframe.to_hdf(path_or_buf=output_path+"/t1_" + file_name[:-4] + ".h5", key="dataset", mode="w", table=True)


