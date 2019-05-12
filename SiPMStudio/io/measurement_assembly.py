import os
import json
import SiPMStudio.io.file_settings as file_settings

from SiPMStudio.processing.measurement import MeasurementArray
from SiPMStudio.processing.measurement import Measurement


def output_measurements(path, measurement_array):
    settings_file = os.path.join(path, "settings.json")
    with open(settings_file, "w+") as settings:
        settings_data = json.load(settings)
        if "measurements" not in settings_data:
            settings_data["measurements"] = []
        for measurement in measurement_array:
            measurement_settings = measurement_array
            settings_data["measurements"].append({"fun_name": measurement.function.__name__,
                                                  "settings": measurement.fun_args,
                                                  "post": measurement.post_name,
                                                  "retrieve": measurement.retrieve})
        json.dump(settings_data, settings, indent=4)




