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
from SiPMStudio.processing.transforms import moving_average


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


def normalize_waveforms():


def main():


if __name__ == "main":
    main()