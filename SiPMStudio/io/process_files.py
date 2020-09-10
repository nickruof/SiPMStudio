import sys
import json

from SiPMStudio.core.digitizers import CAENDT5730
from SiPMStudio.processing.processor import Processor
from SiPMStudio.processing.process_data import process_data


def main():

    settings_file = ""
    if len(sys.argv) == 2:
        settings_file = sys.argv[1]
    else:
        print("Provide a settings file!")
        exit(1)

    settings_dict = None
    with open(settings_file, "r") as json_file:
        settings_dict = json.load(json_file)

    digitizer = CAENDT5730()
    processor = Processor()
    short_tau = settings_dict["deconv_params"]["short_tau"][0]
    long_tau = settings_dict["deconv_params"]["long_tau"][0]
    processor.add("baseline_subtract", settings={"degree": 1})
    processor.add("deconvolve_waves", settings={"short_tau": short_tau, "long_tau": long_tau})
    process_data(settings_dict, processor, digitizer, overwrite=True, output_dir=settings_dict["output_path_t2"], chunk=4000, write_size=3)


if __name__ == "__main__":
    main()