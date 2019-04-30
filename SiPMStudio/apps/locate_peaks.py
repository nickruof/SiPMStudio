import sys

import SiPMStudio.core.digitizers as digitizers
from SiPMStudio.analysis.dark import spectrum_peaks


def main():

    file_name = ""
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
    else:
        print("Specify file name!")

    digitizer1 = digitizers.CAENDT5730(df_data=file_name)
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


if __name__ == "__main__":
    main()
