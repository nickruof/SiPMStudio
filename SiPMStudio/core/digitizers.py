from multiprocessing.sharedctypes import Value
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

    def get_dt(self):
        pass


class CAENDT5730(Digitizer):

    def __init__(self, compass="v1", *args, **kwargs):
        self.compass = compass
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

    def get_event_size(self, t0_file, num_entries):
        num_samples = 0
        if self.compass == "v1":
            with open(t0_file, "rb") as file:
                first_event = file.read(24)
                [num_samples] = np.frombuffer(first_event[20:24], dtype=np.uint32)
            return 24 + 2 * num_samples
        elif self.compass == "v2":
            offset = 0
            with open(t0_file, "rb") as file:
                if num_entries > 0:
                    first_event = file.read(25)
                    [num_samples] = np.frombuffer(first_event[21:25], dtype=np.uint32)
                else:
                    offset = 2
                    first_event = file.read(27)
                    [num_samples] = np.frombuffer(first_event[23:27], dtype=np.uint32)
            return 25 + 2 * num_samples + offset # number of bytes / 2
        else:
            raise AttributeError(f"{self.compass}: version not recognized!")

    def get_event(self, event_data_bytes, num_entries):
        if self.compass == "v1":
            self.decoded_values["board"] = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)[0]
            self.decoded_values["channel"] = np.frombuffer(event_data_bytes[2:4], dtype=np.uint16)[0]
            self.decoded_values["timestamp"] = np.frombuffer(event_data_bytes[4:12], dtype=np.uint64)[0]
            self.decoded_values["energy"] = np.frombuffer(event_data_bytes[12:14], dtype=np.uint16)[0]
            self.decoded_values["energy_short"] = np.frombuffer(event_data_bytes[14:16], dtype=np.uint16)[0]
            self.decoded_values["flags"] = np.frombuffer(event_data_bytes[16:20], np.uint32)[0]
            self.decoded_values["num_samples"] = np.frombuffer(event_data_bytes[20:24], dtype=np.uint32)[0]
            self.decoded_values["waveform"] = np.frombuffer(event_data_bytes[24:], dtype=np.uint16)
        elif self.compass == "v2":
            offset = 0
            if num_entries == 0:
                self.decoded_values["header"] = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)
                offset = 2
            self.decoded_values["board"] = np.frombuffer(event_data_bytes[offset:2], dtype=np.uint16)[0]
            self.decoded_values["channel"] = np.frombuffer(event_data_bytes[offset+2:4], dtype=np.uint16)[0]
            self.decoded_values["timestamp"] = np.frombuffer(event_data_bytes[offset+4:12], dtype=np.uint64)[0]
            self.decoded_values["energy"] = np.frombuffer(event_data_bytes[offset+12:14], dtype=np.uint16)[0]
            self.decoded_values["energy_short"] = np.frombuffer(event_data_bytes[offset+14:16], dtype=np.uint16)[0]
            self.decoded_values["flags"] = np.frombuffer(event_data_bytes[offset+16:20], np.uint32)[0]
            self.decoded_values["code"] = np.frombuffer(event_data_bytes[offset+20:21], np.uint8)[0]
            self.decoded_values["num_samples"] = np.frombuffer(event_data_bytes[offset+21:25], dtype=np.uint32)[0]
            self.decoded_values["waveform"] = np.frombuffer(event_data_bytes[offset+25:], dtype=np.uint16)
        else:
            raise AttributeError(f"{self.compass}: version not recognized!")
        return self._assemble_data_row()

    def get_dt(self):
        dt = (1 / self.sample_rate) * 1e9 # in ns
        return dt

    def _assemble_data_row(self):
        timestamp = self.decoded_values["timestamp"]
        energy = self.decoded_values["energy"]
        energy_short = self.decoded_values["energy_short"]
        flags = self.decoded_values["flags"]
        waveform = self.decoded_values["waveform"]
        return [timestamp, energy, energy_short, flags], waveform
