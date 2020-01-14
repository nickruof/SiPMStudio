import os
import json


def create_database(path, file_name):
    output_file = os.path.join(path, file_name)
    if os.path.exists(output_file):
        raise FileExistsError( output_file+" already exists in this path!")
    data = {"Photocurrent Measurements": {"dates": None}}
    with open(output_file, "w+") as settings:
        data = {"runs": [], "waves": []}
        json.dump(data, settings, indent=4)