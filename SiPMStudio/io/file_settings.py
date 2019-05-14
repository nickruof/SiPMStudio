import os
import json


def _exists(x, array):
    for element in array:
        if element == x:
            return True
        else:
            return False


def _find_index(name, array):
    for i, element in enumerate(array):
        if element["name"] == name:
            return i
    raise LookupError(str(name)+"not found in the array")


def create_json(path):
    output_file = os.path.join(path, "settings.json")
    if os.path.exists(output_file):
        raise FileExistsError("settings.json already exists in this path!")

    with open(output_file, "w+") as settings:
        data = {"files": []}
        json.dump(data, settings, indent=4)


def update_json(path, section, file_name,  key, value):
    settings_file = os.path.join(path, "settings.json")
    with open(settings_file, "r") as file:
        data = json.load(file)

    loc = _find_index(file_name, data["files"])
    data[section][loc][key] = value

    with open(settings_file, "w") as file:
        json.dump(data, file, indent=4)


def add_file(path, file_name):
    settings_file = os.path.join(path, "settings.json")
    with open(settings_file, "r") as file:
        data = json.load(file)
    data["files"].append({"name": file_name, "peaks": None, "wave_peaks": None})
    with open(settings_file, "w") as file:
        json.dump(data, file, indent=4)


def read_file(path, file_name):
    settings_file = os.path.join(path, "settings.json")
    with open(settings_file, "r") as file:
        data = json.load(file)
    loc = _find_index(file_name, data["files"])
    return data["files"][loc]


def file_exists(path, file_name):
    settings_file = os.path.join(path, "settings.json")
    with open(settings_file, "r") as file:
        data = json.load(file)
        for element in data["files"]:
            if element["name"] == file_name:
                return True
    return False





