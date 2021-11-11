import os
import glob
import json
import argparse
import warnings

from SiPMStudio.processing.processor import Processor, load_functions
from SiPMStudio.processing.reprocess_data import reprocess_data

def load_functions(file_name, proc_dict, processor):
    for key, file_dict in proc_dict["processes"].items():
        for name in file_dict.keys():
            if name == file_name:
                processor.add(key, file_dict[name])


def reprocess(settings_dict, proc_dict, processor, verbose=False, pattern=None, file_name=None):
    file_path = settings_dict["output_path_t2"]
    file_list = None
    if file_name is None:
        file_list = glob.glob(f"{file_path}/*.h5")
    else:
        file_list = os.path.join(file_path, file_name)

    for output in proc_dict["save_output"]:
        processor.add_to_file(output)
    
    for i, file_name in enumerate(file_list):
        head_dir, tail_name = os.path.split(file_name)
        if pattern is None:
            load_functions(tail_name, proc_dict, processor)
        elif any(bias in tail_name for bias in pattern):
            load_functions(tail_name, proc_dict, processor)
        else:
            continue
        if len(processor.proc_list) > 0:
            reprocess_data(settings_dict, processor, file_name, verbose=verbose)
        else:
            warnings.warn("No reprocessing functions check that file names in "
                          "--settings and file names in --procs match!", UserWarning)
        processor.clear()


def reprocess_files(settings_file, proc_file, verbose=False, pattern=None):

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    proc_dict = None
    with open(proc_file, "r") as json_file:
        proc_dict = json.load(json_file)

    processor = Processor()
    reprocess(settings_dict, proc_dict, processor, verbose=verbose, pattern=pattern)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--procs", help="processor settings file name")
    parser.add_argument("--verbose", help="set verbosity to True or False", type=bool)
    parser.add_argument("--bias", help="bias name in file pattern")
    args = parser.parse_args()

    bias_list = None
    if args.bias is not None:
        bias_list = [str(i) for i in args.bias.split(",")]

    reprocess_files(args.settings, args.procs, verbose=args.verbose, pattern=bias_list)
