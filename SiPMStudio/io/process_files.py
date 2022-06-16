import json
import argparse

from SiPMStudio.processing.processor import Processor, load_functions, load_functions_v2
from SiPMStudio.processing.process_data import process_data


def process_files(settings_file, proc_file, bias=None, overwrite=True, chunk=4000, write_size=2, verbose=True, load_v2=False):

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    proc_dict = None
    with open(proc_file, "r") as json_file:
        proc_dict = json.load(json_file)

    processor = Processor()
    if load_v2:
        load_functions_v2(proc_dict, processor)
    else:
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
    parser.add_argument("--verbose", help="print extra output at runtime", type=bool, default=True)
    parser.add_argument("--load_v2", help="tell the process to use load_functions or load_funcions_v2", default=False)
    args = parser.parse_args()

    bias_list = None
    if args.bias is not None:
        bias_list = [int(i) for i in args.bias.split(",")]

    process_files(args.settings, args.procs, verbose=args.verbose, bias=bias_list, load_v2=args.load_v2)
