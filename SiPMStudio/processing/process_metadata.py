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
        for i, file_name in enumerate(file_names["files"]):
            num_entries = 0
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
                    num_entries += 1
            _output_to_h5file(file_name, settings["file_base_name"],
                              settings["output_path_raw"], np.array(event_rows), file_names["bias"], digitizer)
            _output_per_waveforms(file_name, settings["file_base_name"], settings["output_path_raw"],
                                  np.array(event_rows), np.array(waveform_rows), file_names["channels"][i],
                                  file_names["bias"], digitizer)
            _output_entries(num_entries, file_names["bias"], settings["file_base_name"], settings["output_path_raw"])
    _output_time(time.time() - start)


def _output_per_waveforms(data_file, output_name, output_path, events, waveforms, channel, bias, digitizer):
    destination = os.path.join(output_path, f"raw_{output_name}_{bias}.h5")
    with h5py.File(destination, "a") as output_file:
        output_file.create_dataset(f"/raw/{channel}/energy", data=events.T[1])
        output_file.create_dataset(f"/raw/{channel}/waveforms", data=waveforms)
        if f"/raw/{channel}/wf_len" in output_file[f"/raw/{channel}"].keys():
            output_file.create_dataset(f"/raw/{channel}/wf_len", data=waveforms.shape[1])


def _output_to_h5file(data_file, output_name, output_path, events, bias, digitizer):
    destination = os.path.join(output_path, f"raw_{output_name}_{bias}.h5")
    with h5py.File(destination, "a") as output_file:
        if "timetag" not in output_file.keys():
            output_file.create_dataset(f"timetag", data=events.T[0])
        if "dt" not in output_file.keys():
            output_file.create_dataset(f"dt", data=digitizer.get_dt())
        if "bias" not in output_file.keys():
            output_file.create_dataset("bias", data=float(bias))
        if "adc_to_v" not in output_file.keys():
            output_file.create_dataset("adc_to_v", data=digitizer.v_range/2**digitizer.adc_bitcount)
        if "date" not in output_file.keys():
            output_file.create_dataset("date", data=time.time())


def _output_entries(n_entries, bias, output_name, output_path):
    destination = os.path.join(output_path, f"raw_{output_name}_{bias}.h5")
    with h5py.File(destination, "a") as output_file:
        if "n_events" not in output_file.keys():
            output_file.create_dataset("n_events", data=n_entries)