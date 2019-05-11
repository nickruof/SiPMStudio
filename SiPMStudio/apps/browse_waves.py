import sys
import matplotlib.pyplot as plt

import SiPMStudio.core.digitizers as digitizers
import SiPMStudio.plots.plots as sipm_plt
import SiPMStudio.analysis.dark as sith
from SiPMStudio.processing.transforms import moving_average


def main():

    file_name = ""
    peaks = False

    if len(sys.argv) == 1:
        print("input a file to browse waveforms")
        return None
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
    if len(sys.argv) >= 3:
        file_name = sys.argv[1]
        peaks = True

    digitizer1 = digitizers.CAENDT5730(df_data=file_name)
    digitizer1.v_range = 2.0
    digitizer1.e_cal = 2.0e-15
    params_data = digitizer1.format_data(waves=False)
    waves_data = digitizer1.format_data(waves=True)
    # new_waves_data = moving_average(waves_data, box_size=15)

    retry = True
    again = False

    while retry:
        if peaks:
            min_distance = float(input("guess minimum distance between peaks "))
            min_height = float(input("guess minimum peak height "))
            width = float(input("guess peak widths "))
            sipm_plt.waveform_plots(waves_data, get_peaks=peaks, min_dist=min_distance, min_height=min_height, width=width)
            plt.show()
            plt.figure(2)
            dts = sith.delay_times(params_data, waves_data, min_height, min_distance, width)
            sipm_plt.plot_delay_times(dts, fit=True)
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


if __name__ == "__main__":
    main()
