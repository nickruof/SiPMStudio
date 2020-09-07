import sys
import glob
import json

from SiPMStudio.core.digitizers import CAENDT5730
from SiPMStudio.processing.processor import Processor
from SiPMStudio.processing.process_data import process_data


def main():

    settings_file = ""
    if len(sys.argv) == 2:
        settings_file = sys.argv[1]
    else:
        raise SystemError("Provide a settings file!")

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    file_list = glob.glob(settings_dict["input_path"]+"/*.bin")
    digitizer = CAENDT5730()
    processor = Processor()
    processor.add("baseline_subtract", settings={"degree": 1})
    processor.add("deconvolve_waves", settings={"height_range": [1600, 3000], "min_loc": 1000})
    process_data(settings_dict, processor, digitizer)


if __name__ == "__main__":
    main()