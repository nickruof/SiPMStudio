import os
import time
import tqdm
import h5py
import peakutils
import numpy as np

from SiPMStudio.processing.process_data import _output_time


def process_metadata(settings, digitizer, output_dir=None, verbose=False):

    print("Processing Metadata! ...")
    print("Number of Files to Process: "+str(len(settings["init_info"])))
    print("Output Path: ", settings["output_path_t1"])

    start = time.time()

    for file_name in tqdm.tqdm(settings["init_info"], total=len(settings["init_info"])):
        event_rows = []
        waveform_rows = []
        baseline_rows = []
        event_size = digitizer.get_event_size(file_name["file_name"])
        with open(file_name["file_name"], "rb") as metadata_file:
            event_data_bytes = metadata_file.read(event_size)
            while event_data_bytes != b"":
                event, waveform = digitizer.get_event(event_data_bytes)
                baseline = peakutils.baseline(waveform, deg=2)
                event_rows.append(event)
                waveform_rows.append(waveform)
                baseline_rows.append(baseline)
                event_data_bytes = metadata_file.read(event_size)
        _output_to_h5file(file_name, settings["file_base_name"], settings["output_path_t1"],
                          np.array(event_rows), np.array(waveform_rows), np.array(baseline_rows), digitizer)
    _output_time(time.time() - start)


def _output_to_h5file(data_file, output_name, output_path, events, waveforms, baselines, digitizer):
    destination = os.path.join(output_path, "t1_"+output_name+"_"+str(data_file["bias"])+".h5")
    with h5py.File(destination, "w") as output_file:
        output_file.create_dataset("/raw/timetag", data=events.T[0])
        output_file.create_dataset("/raw/energy", data=events.T[1])
        output_file.create_dataset("/raw/waveforms", data=waveforms)
        output_file.create_dataset("/raw/baselines", data=baselines)
        output_file.create_dataset("bias", data=float(data_file["bias"]))
        output_file.create_dataset("adc_to_v", data=digitizer.v_range/2**digitizer.adc_bitcount)
