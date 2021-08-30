import sys
import json
import argparse

from SiPMStudio.core.digitizers import CAENDT5730
from SiPMStudio.processing.process_metadata import process_metadata


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="settings file name")
    args = parser.parse_args()
    settings_file = args.input

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    digitizer = CAENDT5730()
    process_metadata(settings_dict, digitizer)


if __name__ == "__main__":
    main()