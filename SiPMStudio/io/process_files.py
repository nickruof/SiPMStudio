import json
import argparse

from SiPMStudio.processing.processor import Processor, load_functions
from SiPMStudio.processing.process_data import process_data


def process_files(settings_file, proc_file, bias=None, overwrite=True, chunk=4000, write_size=2, verbose=True):

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    proc_dict = None
    with open(proc_file, "r") as json_file:
        proc_dict = json.load(json_file)

    processor = Processor()
    load_functions(proc_dict, processor)
    process_data(
        settings_dict, 
        processor, bias=bias, overwrite=overwrite, chunk=chunk, 
        write_size=write_size, 
        verbose=verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--procs", help="processor settings file name")
    parser.add_argument("--bias", help="list of biases to process, comma separated", default=None)
    parser.add_argument("--verbose", help="print extra output at runtime", type=bool)
    args = parser.parse_args()

    process_files(args.settings, args.procs, args.bias, verbose=args.verbose)
