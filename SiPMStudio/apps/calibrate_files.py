import os
import sys
import json
import matplotlib.pyplot as plt

import SiPMStudio.core.digitizers as digitizers
import SiPMStudio.io.file_settings as file_settings

from SiPMStudio.analysis.dark import spectrum_peaks
from SiPMStudio.plots.plots import pc_spectrum

import SiPMStudio.plots.plots as sipm_plt
import SiPMStudio.analysis.dark as sith


from SiPMStudio.analysis.dark import spectrum_peaks
from SiPMStudio.analysis.dark import heights
from SiPMStudio.plots.plots import ph_spectrum


def locate_spectrum_peaks(hist_data):
    plt.figure()
    pc_spectrum(hist_array=[hist_data], log=True)
    plt.show()
    plt.close()

    retry = True
    peaks = []
    while retry:
        min_distance = float(input("guess minimum distance between peaks "))
        min_height = float(input("guess minimum peak height "))
        peaks = spectrum_peaks(params_data=params_data["E_SHORT"], n_bins=2000,
                               min_dist=min_distance, min_height=min_height, display=True)
        remove_peaks = input("Remove Peaks? y/n")
        if remove_peaks == "y":
            what_peaks = input("Input peaks to delete!")
            peak_nums = [float(num) for num in what_peaks.split(" ")]
            peaks = np.delete(peaks, peak_nums)
        again = input("do it again! y/n ")
        if again == "y":
            retry = True
        elif again == "n":
            retry = False
        else:
            break
    return peaks


def locate_waveform_peaks(waves_data):
    retry = True
    again = False

    min_distance = 0
    min_height = 0
    width = 0

    peaks = True

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

    peak_heights = sith.heights(waves_data, min_height, min_distance, width)
    peak_specs = locate_spectrum_peaks(peak_heights)
    return peak_specs


def output_to_json(output_dir, file_name, file_type="waves", pc_peaks=None, ph_peaks=None):
    if not os.path.exists(output_dir+"/settings.json"):
        file_settings.create_json(output_dir)
    if file_settings.file_exists(output_dir, file_name, file_type):
        file_settings.update_json(output_dir, file_type, file_name, "pc_peaks", pc_peaks)
        file_settings.update_json(output_dir, file_type, file_name, "ph_peaks", ph_peaks)
    else:
        file_settings.add_file(output_dir, file_name, file_type)
        file_settings.update_json(output_dir, file_type, file_name, "pc_peaks", pc_peaks)
        file_settings.update_json(output_dir, file_type, file_name, "ph_peaks", ph_peaks)


def main():
    file_name = ""
    output_path = ""
    if len(sys.argv) == 3:
        file_name = sys.argv[1]
        output_path = sys.argv[2]
    else:
        print("Specify <file_name> <output_path>!")

    digitizer = digitizers.CAENDT5730(df_data=file_name)
    digitizer.v_range = 2.0
    digitizer.e_cal = 2.0e-15
    params_data = digitizer.format_data(waves=False)
    waves_data = digitizer.format_data(waves=True)

    pulse_charge_peaks = locate_spectrum_peaks(params_data["E_SHORT"])
    pulse_height_peaks = locate_waveform_peaks(waves_data)
    output_to_json(output_path, file_name, "waves", pulse_charge_peaks, pulse_height_peaks)


if __name__ == "main":
    main()
