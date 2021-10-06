import glob
import json
import argparse

from SiPMStudio.processing.processor import Processor, load_functions
from SiPMStudio.processing.reprocess_data import reprocess_data

def load_functions(file_name, proc_dict, processor):
    for key, file_dict in proc_dict["processes"].items():
        for name in file_dict.keys():
            if name == file_name:
                processor.add(key, file_dict[name])

def reprocess(settings_dict, proc_dict, processor):
    file_path = settings_dict["output_path_t2"]
    file_list = glob.glob(f"{file_path}/*.h5")
    for file_name in file_list:
        load_functions(file_name, proc_dict, processor)
        reprocess_data(settings_dict, processor, file_name)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--procs", help="processor settings file name")
    parser.add_argument("--verbose", help="print extra output at runtime", type=bool)
    args = parser.parse_args()

    settings_file = args.settings
    proc_file = args.procs

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    proc_dict = None
    with open(proc_file, "r") as json_file:
        proc_dict = json.load(json_file)

    processor = Processor()
    reprocess(settings_dict, proc_dict, processor)


if __name__ == "__main__":
    main()