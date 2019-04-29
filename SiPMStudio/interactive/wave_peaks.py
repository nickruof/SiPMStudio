import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import SiPMStudio.core.digitizers as digitizers
import SiPMStudio.plots.plots as sipm_plt
from SiPMStudio.calculations.helpers import detect_peaks
from SiPMStudio.analysis.dark import spectrum_peaks

sns.set_style("whitegrid")

path = "/Users/nickruof/legendsw/SiPMStudio/tests/"
file1 = "t1_0_waves_31.csv"
file2 = "t1_0_waves_32.csv"

digitizer1 = digitizers.CAENDT5730(df_data=path+file1)
digitizer1.v_range = 2.0
digitizer1.e_cal = 2.0e-15
params_data = digitizer1.format_data(waves=False)
waves_data = digitizer1.format_data(waves=True)

retry = True
wave_number = 0
while retry:
    min_distance = float(input("guess minimum distance between peaks "))
    min_height = float(input("guess minimum peak height "))
    sipm_plt.plot_waveform(waves_data.iloc[wave_number, :], find_peaks=True, min_dist=min_distance, min_height=min_height)
    again = input("do it again! y/n ")
    wave_number = int(input("input a waveform number "))
    if again == "y":
        retry = True
    elif again == "n":
        retry = False
    else:
        break