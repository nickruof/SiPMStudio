import sys
import matplotlib.pyplot as plt

import SiPMStudio.core.data_loading as iPad
import SiPMStudio.core.digitizers as iPod
import SiPMStudio.core.devices as device

import SiPMStudio.processing.processor as processor
import SiPMStudio.processing.measurement as measure
from SiPMStudio.processing.process_data import process_data
from SiPMStudio.processing.experiment import Experiment

from SiPMStudio.io.file_settings import read_file

import SiPMStudio.plots.plots as sipm_plt
import SiPMStudio.analysis.light as jedi


def attach_path(path, files):
    new_files = []
    for file in files:
        new_files.append(path+file)
    return new_files


file_path = "/Users/nickruof/Documents/LEGEND/SIPM/PDE/ketek_PDE/"
output_path = file_path
dark_sipm_runs = ["run_29.csv", "run_30.csv", "run_31.csv", "run_32.csv"]
dark_sipm_waves = ["waves_29.csv", "waves_30.csv", "waves_31.csv", "waves_32.csv"]

digitizer = iPod.CAENDT5730()
digitizer.adc_bitcount = 14
digitizer.e_cal = 2.0e-15
digitizer.v_range = 2.0
digitizer.sample_rate = 500e6

sipm = device.sipm(name="KETEK", area=9.0e-6)
sipm.bias = [29.0, 30.0, 31.0, 32.0]

proc = processor.Processor()
proc.add(fun_name="adc_to_volts", settings={"digitizer": digitizer})
proc.add(fun_name="baseline_subtract", settings={})
proc.add(fun_name="moving_average", settings={"box_size": 15})
proc.add(fun_name="normalize_waves", settings={"path": file_path, "file_name": None})

input_files = attach_path(file_path, dark_sipm_waves)
# process_data(data_files=input_files, output_dir=output_path, digitizer=digitizer, processor=proc, multiprocess=False)
# dark_sipm_pwaves = attach_path("t1_", dark_sipm_waves)

dark_sipm_runs = attach_path(file_path, dark_sipm_runs)
dark_sipm_waves = attach_path(file_path, dark_sipm_waves)
measurements = measure.MeasurementArray()
measurements.add(fun_name="gain", settings={"digitizer": digitizer, "sipm": sipm, "path": file_path, "file_name": None})
measurements.add(fun_name="cross_talk", settings={"sipm": sipm, "path": file_path, "file_name": None})
# Experiment(files=dark_sipm_runs, measurement_array=measurements, digitizer=digitizer)

wave_measurements = measure.MeasurementArray()
wave_measurements.add(fun_name="dark_count_rate", settings={"sipm": sipm, "path": file_path, "file_name": None})
Experiment(files=dark_sipm_waves, measurement_array=wave_measurements, digitizer=digitizer)
print(sipm.pulse_rate)
print(sipm.dcr_fit)