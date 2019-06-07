from abc import ABC

import SiPMStudio.analysis.dark as sith
import SiPMStudio.analysis.light as jedi


class MeasurementArray(ABC):

    def __init__(self, settings=None):
        self.file = ""
        self.measurement_list = []
        self.settings = {}
        self.digitizer = None
        self.calcs = []
        self.waves = []

        if settings is not None:
            self.settings = settings
            for key in settings:
                self.add(key, settings[key])

    def set_array(self, digitizer):
        for measurement in self.measurement_list:
            if "file_name" in measurement.fun_args.keys():
                measurement.fun_args["file_name"] = self.file
        self.digitizer = digitizer
        self.calcs = digitizer.format_data(waves=False)
        self.waves = digitizer.format_data(waves=True)

    def run(self):
        for measurement in self.measurement_list:
            if isinstance(measurement, Measurement):
                p_result = measurement.process_block(self.calcs, self.waves)
            else:
                raise TypeError("Unknown Measurment type!")

    def add(self, fun_name, settings=None, post_settings=None, retrieve_settings=None):
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings

        if fun_name in dir(sith):
            self.measurement_list.append(
                Measurement(getattr(sith, fun_name), self.settings[fun_name], post_settings, retrieve_settings))
        elif fun_name in dir(jedi):
            self.measurement_list.append(
                Measurement(getattr(jedi, fun_name), self.settings[fun_name], post_settings, retrieve_settings))
        else:
            raise TypeError("ERROR! unknown measurement function: ", fun_name)

    def overwrite(self, fun_name, settings=None):
        if fun_name not in self.settings:
            raise LookupError("Function to overwrite does not exist!")
        else:
            self.settings[fun_name] = settings
            index = _find_measurement_index(self.measurement_list, fun_name)
            self.measurement_list[i].fun_args=settings


def _find_measurement_index(measurement_list, fun_name):
    for i, measurement in enumerate(measurement_list):
        if measurement.__name__ == fun_name:
            return i
    print("No function found matching "+ fun_name)


class Measurement:

    def __init__(self, function, fun_args=None, post_args=None, retrieve_args=None):
        self.function = function
        self.fun_args = fun_args
        self.post_name = None
        self.retrieve = None
        if post_args is not None:
            self.post_name = post_args["name"]
        if retrieve_args is not None:
            self.retrieve = retrieve_args

    def process_block(self, params, waves, utility_belt=None):
        result = self.function(params_data=params, waves_data=waves, **self.fun_args)
        return result




