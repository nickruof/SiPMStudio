import os
import sys
import json
import matplotlib.pyplot as plt

import SiPMStudio.core.digitizers as digitizers
import SiPMStudio.io.file_settings as file_settings

from SiPMStudio.analysis.dark import spectrum_peaks
from SiPMStudio.plots.plots import pc_spectrum


def exists(x, array):
    for element in array:
        if element == x:
            return True
        else:
            return False


def find_index(name, array):
    for i, element in enumerate(array):
        if element["name"] == name:
            return i
    raise LookupError(str(name)+"not found in the array")


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
    params_data = digitizer1.format_data(waves=False)

    plt.figure()
    pc_spectrum(hist_array=params_data["E_SHORT"], log=True)
    plt.show()
    plt.close()

    retry = True
    peaks = []
    while retry:
        min_distance = float(input("guess minimum distance between peaks "))
        min_height = float(input("guess minimum peak height "))
        peaks = spectrum_peaks(params_data=params_data, min_dist=min_distance, min_height=min_height, display=True)
        again = input("do it again! y/n ")
        if again == "y":
            retry = True
        elif again == "n":
            retry = False
        else:
            break

    output_file = os.path.join(output_path, "settings.json")
    output_peaks = [int(peak) for peak in list(peaks)]
    if not os.path.exists(output_file):
        file_settings.create_json(output_path)
    if file_settings.file_exists(output_path, file_name):
        file_settings.update_json(output_path, "files", file_name, "peaks", output_peaks)
    else:
        file_settings.add_file(output_path, file_name)
        file_settings.update_json(output_path, "files", file_name, "peaks", output_peaks)
    # with open(output_file, "w+") as file:
    #    if os.stat(output_file).st_size != 0:
    #        data = json.load(file)
    #        if exists(output_file, data):
    #            loc = find_index(file_name, data["files"])
    #            peaks = [int(peak) for peak in peaks]
    #            data["files"][loc]["peak_locs"] = peaks
    #        else:
    #            peaks = [int(peak) for peak in peaks]
    #            data["files"].append({"name": file_name, "peak_locs": peaks, "wave_peaks": None})
    #    else:
    #        data["files"] = []
    #        peaks = [int(peak) for peak in peaks]
    #        data["files"].append({"name": file_name, "peak_locs": peaks, "wave_peaks": None})
    #    json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
