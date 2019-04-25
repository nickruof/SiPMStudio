import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/")

import SiPMStudio.core.digitizers as digitizers
from SiPMStudio.calculations.helpers import detect_peaks
from SiPMStudio.analysis.dark import spectrum_peaks

sns.set_style("whitegrid")
sys.path.append("/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/")

path = "/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/tests/"
file1 = "t1_0_waves_31.csv"
file2 = "t1_0_waves_32.csv"

digitizer1 = digitizers.CAENDT5730(df_data=path+file1)
digitizer1.v_range = 2.0
digitizer1.e_cal = 2.0e-15
params_data = digitizer1.format_data(waves=False)

retry = True
while retry:
    min_distance = float(input("guess minimum distance between peaks "))
    min_height = float(input("guess minimum peak height "))
    peaks = spectrum_peaks(params_data=params_data, min_dist=min_distance, min_height=min_height, display=True)
    again = input("do it again! y/n ")
    if again == "y":
        retry = True
    elif again == "n":
        retry = False
        print(peaks)
    else:
        break

