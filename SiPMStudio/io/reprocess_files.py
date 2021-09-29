import json
import argparse

from SiPMStudio.processing.processor import Processor, load_functions
from SiPMStudio.processing.reprocess_data import reprocess_data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", help="settings file name")
    parser.add_argument("--procs", help="processor settings file name")
    parser.add_argument("--verbose", help="print extra output at runtime", type=bool)
    args = parser.parse_args()

    settings_file = args.settings
    proc_file = args.procs
    


if __name__ == "__main__":
    main()