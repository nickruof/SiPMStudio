import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sys.path.append("/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/")
from SiPMStudio.core import data_loading
from SiPMStudio.core import devices
from SiPMStudio.core import digitizers

from SiPMStudio.processing import processor
from SiPMStudio.processing import process_data
from SiPMStudio.processing import measurement

file1 = "t1_0_waves_31.csv"
file2 = "t1_0_waves_32.csv"

Digitizer1 = digitizers.CAENDT5730()
Digitizer1.v_range = 2.0
Digitizer1.e_cal = 2.0e-15
Digitizer2 = digitizers.CAENDT5730()
Digitizer2.v_range = 2.0
Digitizer2.e_cal = 2.0e-15

ketek_32 = devices.sipm(name="ketek", area=9e-6)

measures = measurement.measurement_array()
measures.set_array(digitizer1=Digitizer1)
measures.add(fun_name="fit_multi_gauss", settings={})
proc.add(fun_name="adc_to_volts", settings={"digitizer":Digitizer})
proc.add(fun_name="baseline_subtract", settings={})
proc.add(fun_name="moving_average", settings={"box_size":19})
proc.add(fun_name="savgol", settings={"window":27, "order":2})

process_data.ProcessData(data_files=[path, path2], processor=proc, output_dir=output_directory, multiprocess=True, digitizer=Digitizer, chunk=2000)


