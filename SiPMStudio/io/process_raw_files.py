import sys
import json
import argparse

from SiPMStudio.core.digitizers import CAENDT5730
from SiPMStudio.processing.process_metadata import process_metadata


def process_raw_files(settings_file, compass="v1"):

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    digitizer = CAENDT5730()
    process_metadata(settings_dict, digitizer, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--compass", help="compass version (v1 or v2)", default="v1")
    args = parser.parse_args()

    process_raw_files(args.settings, args.compass)
