import sys
import json
import argparse

from SiPMStudio.core.digitizers import CAENDT5730
from SiPMStudio.processing.processor import Processor
from SiPMStudio.processing.process_data import process_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--procs", help="processor settings file name")
    parser.add_argument("--bias", help="list of biases to process, comma separated")
    args = parser.parse_args()
    settings_file = args.settings
    proc_file = args.procs
    bias = None

    if args.bias:
        bias = [int(i) for i in args.bias.split(",")]

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    proc_dict = None
    with open(proc_file, "r") as json_file:
        proc_dict = json.load(json_file)

    digitizer = CAENDT5730()
    processor = Processor()
    process_data(settings_dict, processor, bias=bias, overwrite=True, chunk=4000, write_size=2)


if __name__ == "__main__":
    main()