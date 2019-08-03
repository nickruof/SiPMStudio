from .data_loading import DataLoader

import numpy as np
import pandas as pd


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

    def initialize_data(self):
        if self.df_data is not None:
            self.df_data = self.df_data.rename(index=str, columns={0: "TIMETAG", 1: "ENERGY", 2: "E_SHORT", 3: "FLAGS"})
        else:
            raise LookupError("No Data Loaded!")

    def format_data(self, waves=False, rows=None):
        if self.df_data is None:
            return None
        if rows is None:
            rows = []
        if len(rows) == 2:
            if waves:
                params_frame = self.df_data.iloc[rows[0]:rows[1], :3]
                params_frame.columns = self.parameters[:3]
                waves_frame = self.df_data.iloc[rows[0]:rows[1], 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[rows[0]:rows[1], :3]
                params_frame.columns = self.parameters[:3]
                return params_frame
        elif len(rows) == 1:
            if waves:
                params_frame = self.df_data.iloc[rows[0]:, :3]
                params_frame.columns = self.parameters[:3]
                waves_frame = self.df_data.iloc[rows[0]:, 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[rows[0]:, :3]
                params_frame.columns = self.parameters[:3]
                return params_frame
        else:
            if waves:
                params_frame = self.df_data.iloc[:, :3]
                params_frame.columns = self.parameters[:3]
                waves_frame = self.df_data.iloc[:, 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[:, :3]
                params_frame.columns = self.parameters[:3]
                return params_frame

    def input_settings(self, settings):
        self.id = settings["id"]
        self.v_range = settings["v_range"]
        self.e_cal = settings["e_cal"]
        self.int_window = settings["int_window"]
        self.file_header = "CH_"+str(settings["channel"])+"@"+self.model_name+"_"+str(settings["id"])+"_Data_"

    def get_event_size(self, t0_file):
        with open(t0_file, "rb") as file:
            first_event = file.read(30)
            [num_samples] = np.frombuffer(first_event[26:30], dtype=np.uint32)
        return 30 + 2 * num_samples  # number of bytes / 2

    def get_event(self, event_data_bytes):
        self.decoded_values["board"] = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)[0]
        self.decoded_values["channel"] = np.frombuffer(event_data_bytes[2:4], dtype=np.uint16)[0]
        self.decoded_values["timestamp"] = np.frombuffer(event_data_bytes[4:12], dtype=np.uint64)[0]
        self.decoded_values["energy"] = np.frombuffer(event_data_bytes[12:20], dtype=np.float64)[0]
        self.decoded_values["energy_short"] = np.frombuffer(event_data_bytes[20:22], dtype=np.uint16)[0]
        self.decoded_values["flags"] = np.frombuffer(event_data_bytes[22:26], np.uint32)[0]
        self.decoded_values["num_samples"] = np.frombuffer(event_data_bytes[26:30], dtype=np.uint32)[0]
        self.decoded_values["waveform"] = np.frombuffer(event_data_bytes[30:], dtype=np.uint16)
        return self._assemble_data_row()

    def _assemble_data_row(self):
        timestamp = self.decoded_values["timestamp"]
        energy = self.decoded_values["energy"]
        energy_short = self.decoded_values["energy_short"]
        flags = self.decoded_values["flags"]
        waveform = self.decoded_values["waveform"]
        return [timestamp, energy, energy_short, flags], waveform

    def create_dataframe(self, array):
        waveform_labels = [str(item) for item in list(range(self.decoded_values["num_samples"]))]
        column_labels = self.parameters + waveform_labels
        dataframe = pd.DataFrame(data=array, columns=column_labels, dtype=float)
        return dataframe

    def parse_xml(self, xmlfile):
        pass
