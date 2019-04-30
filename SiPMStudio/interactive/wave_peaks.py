import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import SiPMStudio.core.digitizers as digitizers
import SiPMStudio.plots.plots as sipm_plt
from SiPMStudio.calculations.helpers import detect_peaks
from SiPMStudio.analysis.dark import spectrum_peaks

def key_event(event, fig, time, waveforms):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(plots)

    ax.plot(time, waveforms.iloc[curr_pos, :])
    fig.canvas.draw()

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
    sipm_plt.waveform_plots(waves_data, find_peaks=True, min_dist=min_distance, min_height=min_height, thresh=0)
    again = input("do it again! y/n ")
    if again == "y":
        retry = True
    elif again == "n":
        retry = False
    else:
        break