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

apparatus = measurement.MeasurementArray()
apparatus.set_array(digitizer=Digitizer1)
fit_settings = {"params_data":apparatus.calcs, "min_distance":80, "min_height":1.0e-5}
gain_settings = {"params"}
apparatus.add(fun_name="fit_multi_gauss", settings=fit_settings)
apparatus.add(fun_name="gain", settings= )




