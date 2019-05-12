import os
import json
import SiPMStudio.io.file_settings as file_settings

from SiPMStudio.processing import processor


def read_processor(path, proc):
    settings_file = os.path.join(path, "settings.json")
    with open(settings_file, "a+") as settings:
        settings_data = json.load(settings)
        if "processor" not in settings_data:
            raise LookupError("No processor settings found!")
        for proc_element in settings_data["processor"]:
            proc.add(fun_name=proc_element["name"], settings=proc_element["settings"])


def output_processor(path, proc):
    settings_file = os.path.join(path, "settings.json")
    with open(settings_file, "w+") as settings:
        settings_data = json.load(settings)
        if "processor" not in settings_data:
            settings_data["processor"] = []
        for key, value in proc.settings.iteritems():
            settings_data["processor"].append({"name": key, "settings": value})
        settings_data.dump(settings_data, settings_file, indent=4)
