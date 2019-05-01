import matplotlib.pyplot as plt

import SiPMStudio.core.data_loading as iPad
import SiPMStudio.core.digitizers as iPod
import SiPMStudio.core.devices as device

import SiPMStudio.processing.processor as processor
import SiPMStudio.processing.measurement as measure

import SiPMStudio.plots.plots as sipm_plt

from SiPMStudio.processing.process_data import ProcessData
from SiPMStudio.processing.experiment import Experiment

def attach_path(path, files):
    new_files = []
    for file in files:
        new_files.append(path+file)
    return new_files

file_path = "/Users/nickruof/Documents/LEGEND/SIPM/PDE/ketek_PDE/"
output_path = file_path
dark_sipm_runs = ["run_29.csv", "run_30.csv", "run_31.csv", "run_32.csv"]
dark_sipm_waves = ["wave_29.csv", "wave_30.csv", "wave_31.csv", "wave_32.csv"]
dark_sipm_I = ["dark29.csv", "dark30.csv", "dark31.csv", "dark32.csv"]
light_sipm_I = ["lit29.csv", "lit30.csv", "lit31.csv", "lit32.csv"]

digitizer = iPod.CAENDT5730()
digitizer.adc_bitcount = 14
digitizer.e_cal = 2.0e-15
digitizer.v_range = 2.0
digitizer.sample_rate = 500e6

scope = iPad.Keithley2450()

sipm = device.sipm(name="KETEK", area=9.0e-6)
sipm.bias = [29.0, 30.0, 31.0, 32.0]
diode = device.photodiode(name="SM05PD1B", area=1.296e-5)
diode.load_response(file_path=file_path+"responsivity.csv")

proc = processor.Processor()
proc.add(fun_name="adc_to_volts", settings={"digitizer": digitizer})
proc.add(fun_name="baseline_subtract", settings={})
proc.add(fun_name="moving_average", settings={"box_size": 19})
proc.add(fun_name="savgol", settings={"window": 51, "order": 3})

# input_files = attach_path(file_path, dark_sipm_waves)
# ProcessData(data_files=input_files, output_dir=output_path, digitizer=digitizer, processor=processor, multiprocess=True)
# dark_sipm_pwaves = attach_path("t1_", dark_sipm_waves)

measurements = []
for bias in sipm.bias:
    measurements.append(measure.MeasurementArray())

belt = measure.UtilityBelt()
belt.set_belt(names=["peak_locations", "dts"])
# posted data settings
fit_post = {"name": "peak_locations"}
dt_post = {"name": "dts"}
# retrieve data from post settings
gain_retrieve = {"variable": "peaks", "name": "peak_locations"}
dt_retrieve = {"variable": "dts", "name": "dts"}
peak_settings = [
    {"min_dist": 60, "min_height": 0.0015},
    {"min_dist": 70, "min_height": 0.0015},
    {"min_dist": 70, "min_height": 0.0015},
    {"min_dist": 70, "min_height": 0.0015}
]
gain_settings = {"digitizer": digitizer, "sipm": sipm}
measure.recursive_add(measurement_arrays=measurements,
                      fun_name="spectrum_peaks", settings=peak_settings, post_settings=fit_post)
measure.global_add(measurement_arrays=measurements,
                   fun_name="gain", settings=gain_settings, retrieve_settings=gain_retrieve)
measure.global_add(measurement_arrays=measurements,
                   fun_name="cross_talk", settings={"sipm": sipm}, retrieve_settings=gain_retrieve)

input_files = attach_path(file_path, dark_sipm_runs)
Experiment(files=input_files, measurement_arrays=measurements, utility_belt=belt, digitizer=digitizer)

plt.close()
sipm_plt.plot_gain(sipm)
sipm_plt.plot_cross_talk(sipm)


