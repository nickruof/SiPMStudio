import os
import glob
import time
import h5py
import numpy as np

from SiPMStudio.processing.process_data import _output_time
from SiPMStudio.utils.gen_utils import tqdm_it
import SiPMStudio.processing.calculators as pc


def process_metadata(settings, digitizer, overwrite=True, verbose=False):

    if verbose:
        print("Processing Metadata! ...")
        print("Number of Files to Process: "+str(len(settings["init_info"])))
        print("Output Path: ", settings["output_path_raw"])

    if not os.path.exists(settings["output_path_raw"]):
        os.makedirs(settings["output_path_raw"])

    if overwrite is True:
        output_path = settings["output_path_raw"]
        file_list = glob.glob(f"{output_path}/*.h5")
        for file in file_list:
            os.remove(file)

    start = time.time()
    for file_names in tqdm_it(settings["init_info"], verbose=verbose):
        for i, file_name in enumerate(file_names["files"]):
            num_entries = 0
            event_rows = []
            waveform_rows = []
            first_event_size, event_size = digitizer.get_event_size(file_name)
            with open(file_name, "rb") as metadata_file:
                event_data_bytes = metadata_file.read(first_event_size)
                while event_data_bytes != b"":
                    event, waveform = digitizer.get_event(event_data_bytes, num_entries)
                    event_rows.append(event)
                    waveform_rows.append(waveform)
                    event_data_bytes = metadata_file.read(event_size)
                    num_entries += 1
            output_name = settings["file_base_name"]
            if "test" in file_names.keys():
                tag = file_names["test"]
                output_name += f"_{tag}"
            destination = _output_to_h5file(output_name,
                              settings["output_path_raw"], np.array(event_rows),
                              file_names["bias"], digitizer)
            _output_per_waveforms(destination,
                                  np.array(event_rows), np.array(waveform_rows), file_names["channels"][i],
                                  file_names["bias"], settings["amplifier"], digitizer, settings["v_range"])
            _output_entries(num_entries, destination)
    _output_time(time.time() - start)


def process_metadata_csv(settings, digitizer, overwrite=True, verbose=False):

    if verbose:
        print("Processing Metadata! ...")
        print("Number of Files to Process: "+str(len(settings["init_info"])))
        print("Output Path: ", settings["output_path_raw"])

    if not os.path.exists(settings["output_path_raw"]):
        os.makedirs(settings["output_path_raw"])

    if overwrite is True:
        output_path = settings["output_path_raw"]
        file_list = glob.glob(f"{output_path}/*.h5")
        for file in file_list:
            os.remove(file)

    start = time.time()
    for file_names in tqdm_it(settings["init_info"], verbose=verbose):
        for i, file_name in enumerate(file_names["files"]):
            num_entries = 0
            event_rows = []
            waveform_rows = []
            first_event_size, event_size = digitizer.get_event_size_csv(file_name)
            with open(file_name, "r") as metadata_file:
                full_data = metadata_file.readlines(first_event_size)
                data_rows = [int(i, 0) for i in full_data.split(";")]
                for i, data_line in enumerate(full_data[1:]):
                    data_elements = [int(i, 0) for i in data_line.split(";")]
                    event, waveform = digitizer.get_event_csv(data_elements)
                    event_rows.append(event)
                    waveform_rows.append(waveform)
                    num_entries += 1
            output_name = settings["file_base_name"]
            if "test" in file_names.keys():
                tag = file_names["test"]
                output_name += f"_{tag}"
            destination = _output_to_h5file(output_name,
                              settings["output_path_raw"], np.array(event_rows),
                              file_names["bias"], digitizer)
            _output_per_waveforms(destination,
                                  np.array(event_rows), np.array(waveform_rows), file_names["channels"][i],
                                  file_names["bias"], settings["amplifier"], digitizer, settings["v_range"])
            _output_entries(num_entries, destination)
    _output_time(time.time() - start)


def _output_per_waveforms(destination, events, waveforms, channel, bias, amplifier, digitizer, v_range):
    with h5py.File(destination, "a") as output_file:
        output_file.create_dataset(f"/raw/channels/{channel}/energy", data=events.T[1])
        output_file.create_dataset(f"/raw/channels/{channel}/waveforms", data=waveforms)
        if "wf_len" not in output_file[f"/raw/channels/{channel}"].keys():
            output_file.create_dataset(f"/raw/channels/{channel}/wf_len", data=waveforms.shape[1])
        if "amp" not in output_file[f"/raw/channels/{channel}"].keys():
            amplification = _compute_amplification(amplifier, channel)
            output_file.create_dataset(f"/raw/channels/{channel}/amp", data=amplification)
        if "adc_to_v" not in output_file[f"/raw/channels/{channel}"].keys():
            output_file.create_dataset(f"/raw/channels/{channel}/adc_to_v", data=v_range[channel]/2**digitizer.adc_bitcount)


def _output_to_h5file(output_name, output_path, events, bias, digitizer):
    destination = os.path.join(output_path, f"raw_{output_name}_{bias}.h5")
    with h5py.File(destination, "a") as output_file:
        if "timetag" not in output_file.keys():
            output_file.create_dataset(f"timetag", data=events.T[0] / 1e3)
        if "dt" not in output_file.keys():
            output_file.create_dataset(f"dt", data=digitizer.get_dt())
        if "bias" not in output_file.keys():
            output_file.create_dataset("bias", data=float(bias))
        if "date" not in output_file.keys():
            output_file.create_dataset("date", data=time.time())
    return destination


def _output_entries(n_entries, destination):
    with h5py.File(destination, "a") as output_file:
        if "n_events" not in output_file.keys():
            output_file.create_dataset("n_events", data=n_entries)


def _compute_amplification(settings, channel):
    full_amp = 1
    if channel not in settings.keys():
        return full_amp
    for i, func in enumerate(settings[channel]["functions"]):
        amp_func = getattr(pc, func)
        full_amp *= amp_func(**settings[channel]["values"][i])
    return full_amp
