import os
import time
import h5py
import numpy as np

from SiPMStudio.processing.process_data import _output_time
from SiPMStudio.utils.gen_utils import tqdm_it


def process_metadata(settings, digitizer, output_dir=None, verbose=False):

    if verbose:
        print("Processing Metadata! ...")
        print("Number of Files to Process: "+str(len(settings["init_info"])))
        print("Output Path: ", settings["output_path_raw"])

    start = time.time()

    for file_names in tqdm_it(settings["init_info"], verbose=verbose):
        for i, file_name in enumerate(file_names):
            event_rows = []
            waveform_rows = []
            event_size = digitizer.get_event_size(file_name["file_name"])
            with open(file_name["file_name"], "rb") as metadata_file:
                event_data_bytes = metadata_file.read(event_size)
                while event_data_bytes != b"":
                    event, waveform = digitizer.get_event(event_data_bytes)
                    event_rows.append(event)
                    waveform_rows.append(waveform)
                    event_data_bytes = metadata_file.read(event_size)
            _output_to_h5file(file_name, settings["file_base_name"],
                              settings["output_path_raw"], np.array(event_rows), digitizer)
            _output_per_waveforms(file_name, settings["file_base_name"], settings["output_path_raw"],
                            np.array(event_rows), np.array(waveform_rows), file_names["channels"][i], digitizer)
    _output_time(time.time() - start)


def _output_per_waveforms(data_file, output_name, output_path, events, waveforms, channel, digitizer):
    bias = data_file["bias"]
    destination = os.path.join(output_path, f"raw_{output_name}_{bias}.h5")
    with h5py.File(destination, "w") as output_file:
        output_file.create_dataset(f"/raw/{channel}/energy", data=events.T[1])
        output_file.create_dataset(f"/raw/{channel}/waveforms", data=waveforms)


def _output_to_h5file(data_file, output_name, output_path, events, digitizer):
    bias = data_file["bias"]
    destination = os.path.join(output_path, f"raw_{output_name}_{bias}.h5")
    with h5py.File(destination, "w") as output_file:
        if "/raw/timetag" not in output_file.keys():
            output_file.create_dataset(f"/raw/timetag", data=events.T[0])
        if "/raw/dt" not in output_file.keys():
            output_file.create_dataset(f"/raw/dt", data=digitizer.get_dt())
        if "bias" not in output_file.keys():
            output_file.create_dataset("bias", data=float(data_file["bias"]))
        if "adc_to_v" not in output_file.keys():
            output_file.create_dataset("adc_to_v", data=digitizer.v_range/2**digitizer.adc_bitcount)
        if "date" not in output_file.keys():
            output_file.create_dataset("date", data=time.time())
