import sys
import matplotlib.pyplot as plt

import SiPMStudio.core.data_loading as iPad
import SiPMStudio.core.digitizers as iPod
import SiPMStudio.core.devices as device

import SiPMStudio.processing.processor as processor
import SiPMStudio.processing.measurement as measure

import SiPMStudio.plots.plots as sipm_plt
import SiPMStudio.analysis.light as jedi

from SiPMStudio.processing.process_data import process_data
from SiPMStudio.processing.experiment import Experiment


def attach_path(path, files):
    new_files = []
    for file in files:
        new_files.append(path+file)
    return new_files


file_path = "/Users/nickruof/Documents/LEGEND/SIPM/PDE/ketek_PDE/"
output_path = file_path
dark_sipm_runs = ["run_29.csv", "run_30.csv", "run_31.csv", "run_32.csv"]
dark_sipm_waves = ["waves_29.csv", "waves_30.csv", "waves_31.csv", "waves_32.csv"]
dark_I = ["diodedark.csv", "dark29.csv", "dark30.csv", "dark31.csv", "dark32.csv"]
light_I = ["diodelit.csv", "lit29.csv", "lit30.csv", "lit31.csv", "lit32.csv"]

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
led = device.led(name="green_led", wavelength=5.75e-7)

proc = processor.Processor()
proc.add(fun_name="adc_to_volts", settings={"digitizer": digitizer})
proc.add(fun_name="baseline_subtract", settings={})
proc.add(fun_name="moving_average", settings={"box_size": 15})

input_files = attach_path(file_path, dark_sipm_waves)
process_data(data_files=input_files, output_dir=output_path, digitizer=digitizer, processor=proc, multiprocess=False)
dark_sipm_pwaves = attach_path("t1_", dark_sipm_waves)

sys.exit()

########################################
# Spectrum Measurements without waves ###
########################################

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

############################
# Measurements with waves ###
############################

measurements = []
for bias in sipm.bias:
    measurements.append(measure.MeasurementArray())

belt = measure.UtilityBelt()
belt.set_belt(names=["dts"])
dt_post = {"name": "dts"}
dt_retrieve = {"variable": "dts", "name": "dts"}
dt_settings = [
    {"min_dist": 50, "min_height": 8155, "width": 4},
    {"min_dist": 50, "min_height": 8155, "width": 4},
    {"min_dist": 50, "min_height": 8155, "width": 4},
    {"min_dist": 50, "min_height": 8155, "width": 4},
]
measure.recursive_add(measurement_arrays=measurements,
                      fun_name="delay_times", settings=dt_settings, post_settings=dt_post)
measure.global_add(measurement_arrays=measurements,
                   fun_name="dcr_exp_fit", settings={"sipm": sipm}, retrieve_settings=dt_retrieve)
measure.global_add(measurement_arrays=measurements,
                   fun_name="pulse_rate", settings={"sipm": sipm, "min_dist": 50, "min_height": 8155, "width": 4})

input_files = attach_path(file_path, dark_sipm_waves)
Experiment(files=input_files, measurement_arrays=measurements, utility_belt=belt, digitizer=digitizer)

dark_files = attach_path(file_path, dark_I)
light_files = attach_path(file_path, light_I)

jedi.continuous_pde(dataloader=scope, sipm=sipm, diode=diode,
                    led=led, bias=sipm.bias, dark_files=dark_files, light_files=light_files)


plt.close()
plt.figure(1)
sipm_plt.plot_gain(sipm, lin_fit=True)
plt.figure(2)
sipm_plt.plot_cross_talk(sipm)
plt.figure(3)
sipm_plt.plot_pde(sipm)
plt.show()



