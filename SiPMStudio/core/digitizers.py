from .data_loading import DataLoader

import numpy as np


class Digitizer(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format_data(self, waves=False, rows=None):
        pass

    def get_event_size(self, t0_file):
        pass

    def get_event(self, event_data_bytes):
        pass


class CAENDT5730(Digitizer):

    def __init__(self, *args, **kwargs):
        self.id = None
        self.model_name = "DT5730"
        self.file_header = None
        self.adc_bitcount = 14
        self.sample_rate = 500e6
        self.v_range = 2.0

        self.e_cal = None
        self.int_window = None
        self.parameters = ["TIMETAG", "ENERGY", "E_SHORT", "FLAGS"]

        self.decoded_values = {
            "board": None,
            "channel": None,
            "timestamp": None,
            "energy": None,
            "energy_short": None,
            "flags": None,
            "num_samples": None,
            "waveform": []
        }
        super().__init__(*args, **kwargs)

    def input_settings(self, settings):
        self.id = settings["id"]
        self.v_range = settings["v_range"]
        self.e_cal = settings["e_cal"]
        self.int_window = settings["int_window"]
        self.file_header = "DataR_CH"+str(settings["channel"])+"@"+self.model_name+"_"+str(settings["id"])+"_"

    def get_event_size(self, t0_file):
        with open(t0_file, "rb") as file:
            first_event = file.read(24)
            [num_samples] = np.frombuffer(first_event[20:24], dtype=np.uint32)
        return 24 + 2 * num_samples  # number of bytes / 2

    def get_event(self, event_data_bytes):
        self.decoded_values["board"] = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)[0]
        self.decoded_values["channel"] = np.frombuffer(event_data_bytes[2:4], dtype=np.uint16)[0]
        self.decoded_values["timestamp"] = np.frombuffer(event_data_bytes[4:12], dtype=np.uint64)[0]
        self.decoded_values["energy"] = np.frombuffer(event_data_bytes[12:14], dtype=np.uint16)[0]
        self.decoded_values["energy_short"] = np.frombuffer(event_data_bytes[14:16], dtype=np.uint16)[0]
        self.decoded_values["flags"] = np.frombuffer(event_data_bytes[16:20], np.uint32)[0]
        self.decoded_values["num_samples"] = np.frombuffer(event_data_bytes[20:24], dtype=np.uint32)[0]
        self.decoded_values["waveform"] = np.frombuffer(event_data_bytes[24:], dtype=np.uint16)
        return self._assemble_data_row()

    def _assemble_data_row(self):
        timestamp = self.decoded_values["timestamp"]
        energy = self.decoded_values["energy"]
        energy_short = self.decoded_values["energy_short"]
        flags = self.decoded_values["flags"]
        waveform = self.decoded_values["waveform"]
        return [timestamp, energy, energy_short, flags], waveform
