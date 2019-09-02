import os
import sys
import numpy as np
from uncertainties import unumpy
import animation
import matplotlib.pyplot as plt
import pickle as pk

import SiPMStudio.core.digitizers as digitizers
import SiPMStudio.io.file_settings as file_settings

from SiPMStudio.processing.processor import Processor
from SiPMStudio.processing.process_data import process_data

import SiPMStudio.analysis.dark as sith
from SiPMStudio.analysis.dark import spectrum_peaks
from SiPMStudio.analysis.dark import triggered_heights

from SiPMStudio.plots.plotting import pc_spectrum
import SiPMStudio.plots.plotting as sipm_plt


def locate_spectrum_peaks(hist_data, bin_width=1.0, file_name=None, output_path=None):
    bins = int(round((max(hist_data)-min(hist_data)) / bin_width))
    fig, ax = plt.subplots()
    [n_hist, bin_edges] = pc_spectrum(ax, hist_array=[hist_data], n_bins=bins, log=True)
    plt.show()

    retry = True
    peaks = []
    while retry:
        min_distance = float(input("guess minimum distance between peaks "))
        min_height = float(input("guess minimum peak height "))
        peaks = spectrum_peaks(params_data=hist_data, n_bins=bins,
                               min_dist=min_distance, min_height=min_height, display=True)
        again = input("do it again! y/n ")
        if again == "y":
            retry = True
        elif again == "n":
            retry = False
            try:
                peaks = spectrum_peaks(params_data=hist_data, n_bins=bins,
                                       min_dist=min_distance, min_height=min_height, display=True, fit_peaks=True)
            except RuntimeError:
                again = input("Fits did not Converge, try again? y/n ")
                if again == "y":
                    retry = True
                else:
                    peak_inputs = input("Input peak locations manually! ")
                    peak_errors = input("Input estimated uncertainties! ")
                    peak_locs = np.array([int(num) for num in peak_inputs.split(" ")])
                    errors = np.array([int(num) for num in peak_errors.split(" ")])
                    peaks = unumpy.uarray(peak_locs, errors)
            remove_peaks = input("Remove Peaks? y/n ")
            if remove_peaks == "y":
                what_peaks = input("Input peaks to delete! ")
                peak_nums = [int(num) for num in what_peaks.split(" ")]
                peaks = np.delete(peaks, peak_nums)
        else:
            break
    if (file_name is not None) and (output_path is not None):
        pk.dump(peaks, open(output_path+"/"+file_name[:-3].replace("t1", "t2")+"_heights.pk", "wb"))
    return peaks


def locate_triggered_peaks(waves_data, bin_width=1):
    fig, ax = plt.subplots()
    sipm_plt.plot_waveforms(ax, waves_data.iloc[:, 0:100])
    plt.show()
    index = int(input("Input a trigger index location: "))
    heights = triggered_heights(waves_data, index)
    peak_locs = locate_spectrum_peaks(heights, bin_width)
    return peak_locs


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
    peak_specs = locate_spectrum_peaks(peak_heights, 400)

    return peak_specs


def output_to_json(output_dir, file_name, file_type="waves", pc_peaks=None, ph_peaks=None):
    store_pc = [float(x_peak) for x_peak in pc_peaks]
    store_ph = [float(x_peak) for x_peak in ph_peaks]
    if not os.path.exists(output_dir+"/settings.json"):
        file_settings.create_json(output_dir)
    if file_settings.file_exists(output_dir, file_name, file_type):
        file_settings.update_json(output_dir, file_type, file_name, "pc_peaks", store_pc)
        file_settings.update_json(output_dir, file_type, file_name, "ph_peaks", store_ph)
    else:
        file_settings.add_file(output_dir, file_name, file_type)
        file_settings.update_json(output_dir, file_type, file_name, "pc_peaks", store_pc)
        file_settings.update_json(output_dir, file_type, file_name, "ph_peaks", store_ph)


def main():
    input_path = ""
    file_name = ""
    output_path = ""

    if len(sys.argv) == 4:
        file_name = sys.argv[1]
        input_path = os.path.abspath(sys.argv[2])
        output_path = os.path.abspath(sys.argv[3])
    elif len(sys.argv) == 3:
        file_name = sys.argv[1]
        output_path = os.path.abspath(sys.argv[2])
        input_path = os.getcwd()
    else:
        print("Specify <file_name> <output_path>!")

    if not os.path.isfile(file_name):
        raise FileNotFoundError("File: "+str(file_name)+" not found!")

    wait = animation.Wait(text="Loading File: "+file_name+" ")
    print(" ")
    wait.start()
    digitizer = digitizers.CAENDT5730(df_data=file_name)
    digitizer.v_range = 2.0
    digitizer.e_cal = 2.5e-15
    params_data = digitizer.format_data(waves=False)
    waves_data = digitizer.format_data(waves=True)
    wait.stop()
    print(" ")

    pulse_charge_peaks = locate_spectrum_peaks(params_data["ENERGY"], 1, file_name, output_path)
    pulse_height_peaks = locate_triggered_peaks(waves_data)

    norm_proc = Processor()
    norm_proc.add(fun_name="normalize_waves", settings={"peak_locs": unumpy.nominal_values(pulse_height_peaks)})
    norm_proc.add(fun_name="baseline_subtract", settings={})
    norm_proc.add(fun_name="normalize_energy", settings={"pc_peaks": unumpy.nominal_values(pulse_charge_peaks), "label": "ENERGY"})
    t1_file = file_name
    t1_path = input_path

    include_other_files = input("Include other files to process? y/n? ")
    file_list = [t1_file]
    if include_other_files == "y":
        what_files = input("Provide file names: ")
        for file in what_files.split(" "):
            file_list.append(file)

    process_data(t1_path, file_list, norm_proc, digitizer, output_dir=output_path, overwrite=False, write_size=5)


if __name__ == "__main__":
    main()
