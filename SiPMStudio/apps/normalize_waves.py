import os
import sys
import matplotlib.pyplot as plt

import SiPMStudio.core.digitizers as digitizers
import SiPMStudio.io.file_settings as file_settings

from SiPMStudio.analysis.dark import spectrum_peaks
from SiPMStudio.analysis.dark import heights
from SiPMStudio.plots.plots import ph_spectrum


def exists(x, array):
    for element in array:
        if element == x:
            return True
        else:
            return False


def main():

    file_name = ""
    output_path = ""
    if len(sys.argv) == 3:
        file_name = sys.argv[1]
        output_path = sys.argv[2]
    else:
        print("Specify <file_name> <output_path>!")

    digitizer1 = digitizers.CAENDT5730(df_data=file_name)
    digitizer1.v_range = 2.0
    digitizer1.e_cal = 2.0e-15
    waves_data = digitizer1.format_data(waves=True)
    file_type = "waves"
    settings = file_settings.read_file(output_path, file_name, file_type)["wave_peaks"]
    peak_heights = heights(waves_data, settings["min_height"], settings["min_dist"], settings["width"])

    plt.figure()
    ph_spectrum(heights=peak_heights, log=True)
    plt.show()
    plt.close()

    retry = True
    peaks = []
    while retry:
        min_distance = float(input("guess minimum distance between peaks "))
        min_height = float(input("guess minimum peak height "))
        peaks = spectrum_peaks(params_data=peak_heights, min_dist=min_distance, min_height=min_height, display=True)
        again = input("do it again! y/n ")
        if again == "y":
            retry = True
        elif again == "n":
            retry = False
        else:
            break

    output_file = os.path.join(output_path, "settings.json")
    output_peaks = [float(peak) for peak in list(peaks)]
    if not os.path.exists(output_file):
        file_settings.create_json(output_path)
    if file_settings.file_exists(output_path, file_name, file_type):
        file_settings.update_json(output_path, file_type, file_name, "height_peaks", output_peaks)
    else:
        file_settings.add_file(output_path, file_name)
        file_settings.update_json(output_path, file_type, file_name, "height_peaks", output_peaks)


if __name__ == "__main__":
    main()
