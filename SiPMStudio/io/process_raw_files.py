import sys
import glob
import json

from SiPMStudio.core.digitizers import CAENDT5730
from SiPMStudio.processing.process_metadata import process_metadata


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
    process_metadata(settings_dict, digitizer)


if __name__ == "__main__":
    main()