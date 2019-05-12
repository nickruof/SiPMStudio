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
    if os.stat(output_file).st_size != 0:
        raise FileExistsError("settings.json already exists in this path!")

    with open(output_file, "w+") as settings:
        data = {"files": []}
        json.dump(data, settings, indent=4)


def update_json(path, file_name, section, key, value):
    settings_file = os.path.join(path, "settings.json")
    with open(settings_file, "w") as file:
        data = json.load(file)
        loc = _find_index(file_name, data["files"])
        data[section][loc][key] = value
        json.dump(data, settings, indent=4)



