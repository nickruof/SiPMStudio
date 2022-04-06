import sys
import json
import argparse

from SiPMStudio.core.digitizers import CAENDT5730
from SiPMStudio.processing.process_metadata import process_metadata, process_metadata_csv


def process_raw_files(settings_file, compass="v1", csv=False):

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    digitizer = CAENDT5730(compass=compass)
    if csv is True:
        process_metadata_csv(settings_dict, digitizer, verbose=True)
    else:
        process_metadata(settings_dict, digitizer, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--compass", help="compass version (v1 or v2)", default="v1")
    parser.add_argument("--csv", help="set True if using csv input files", type=bool, default=False)
    args = parser.parse_args()

    process_raw_files(args.settings, args.compass, args.csv)
