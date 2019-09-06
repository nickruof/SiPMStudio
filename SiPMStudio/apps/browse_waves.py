import os
import sys
import animation
import matplotlib.pyplot as plt

import SiPMStudio.core.digitizers as digitizers
import SiPMStudio.plots.plotting as sipm_plt
import SiPMStudio.analysis.dark as sith
import SiPMStudio.io.file_settings as file_settings
from SiPMStudio.processing.transforms import moving_average


def _exists(x, array):
    for element in array:
        if element == x:
            return True
        else:
            return False


def main():

    file_name = ""
    output_dir = ""
    peaks = False

    if len(sys.argv) == 1:
        print("input a file to browse waveforms")
        return None
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
        peaks = True
    if len(sys.argv) >= 3:
        file_name = sys.argv[1]
        output_dir = sys.argv[2]
        peaks = True

    wait = animation.Wait(text="Loading File: " + file_name + " ")
    print(" ")
    wait.start()
    digitizer1 = digitizers.CAENDT5730(df_data=file_name)
    digitizer1.v_range = 2.0
    digitizer1.e_cal = 2.0e-15
    waves_data = digitizer1.format_data(waves=True)
    print(" ")
    wait.stop()

    retry = True
    again = False

    min_distance = 0
    min_height = 0
    width = 0

    heights = []
    while retry:
        if peaks:
            min_distance = float(input("guess minimum distance between peaks "))
            min_height = float(input("guess minimum peak height "))
            width = float(input("guess peak widths "))
            sipm_plt.waveform_plots(waves_data, get_peaks=peaks, min_dist=min_distance, min_height=min_height, width=width)
            plt.show()
            again = input("do it again! y/n ")
        else:
            sipm_plt.waveform_plots(waves_data, get_peaks=peaks)
            plt.show()
            again = input("do it again! y/n ")
        if again == "y":
            retry = True
        elif again == "n":
            retry = False
        else:
            break

    # heights = sith.heights(waves_data, min_height, min_distance, width)
    # plt.figure(2)
    # sipm_plt.ph_spectrum(heights, log=True)
    # plt.show()


if __name__ == "__main__":
    main()
