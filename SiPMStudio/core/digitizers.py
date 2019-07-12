from .data_loading import DataLoader

import numpy as np
import pandas as pd


class Digitizer(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_event = None

    def format_data(self, waves=False, rows=None):
        pass


class CAENDT5730(Digitizer):

    def __init__(self, *args, **kwargs):
        self.id = None
        self.model_name = "CAENDT5730"
        self.adc_bitcount = 14
        self.sample_rate = 500e6
        self.v_range = 2.0

        self.e_cal = None
        self.int_window = None
        self.parameters = ["TIMETAG", "E_LONG", "E_SHORT"]

        self.event_size = 0
        self.metadata_event = {
            "board": None,
            "channel": None,
            "timestamp": None,
            "energy": None,
            "energy_short": None,
            "flags": None,
            "num_samples": None,
            "waveform": None
        }
        super().__init__(*args, **kwargs)

    def initialize_data(self):
        if self.df_data is not None:
            self.df_data = self.df_data.rename(index=str, columns={0: "TIMETAG", 1: "E_LONG", 2: "E_SHORT", 3: "FLAGS"})
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
                params_frame.columns = self.parameters
                waves_frame = self.df_data.iloc[rows[0]:rows[1], 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[rows[0]:rows[1], :3]
                params_frame.columns = self.parameters
                return params_frame
        elif len(rows) == 1:
            if waves:
                params_frame = self.df_data.iloc[rows[0]:, :3]
                params_frame.columns = self.parameters
                waves_frame = self.df_data.iloc[rows[0]:, 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[rows[0]:, :3]
                params_frame.columns = self.parameters
                return params_frame
        else:
            if waves:
                params_frame = self.df_data.iloc[:, :3]
                params_frame.columns = self.parameters
                waves_frame = self.df_data.iloc[:, 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[:, :3]
                params_frame.columns = self.parameters
                return params_frame

    def input_settings(self, settings):
        self.id = settings["id"]
        self.v_range = settings["v_range"]
        self.e_cal = settings["e_cal"]
        self.int_window = settings["int_window"]

    def get_event_size(self, t0_file):
        with open(t0_file, "rb") as file:
            first_event = file.read(24)
            [num_samples] = np.frombuffer(first_event[20:24], dtype=np.uint32)
            self.event_size = 24 + 2 * num_samples  # number of bytes / 2

    def get_event(self, event_data_bytes):
        self.metadata_event["board"] = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)
        self.metadata_event["channel"] = np.frombuffer(event_data_bytes[2:4], dtype=np.uint16)
        self.metadata_event["timestamp"] = np.frombuffer(event_data_bytes[4:12], dtype=np.uint64)
        self.metadata_event["energy"] = np.frombuffer(event_data_bytes[12:14], dtype=np.uint16)
        self.metadata_event["energy_short"] = np.frombuffer(event_data_bytes[14:16], dtype=np.uint16)
        self.metadata_event["flags"] = np.frombuffer(event_data_bytes[16:20], np.uint32)
        self.metadata_event["num_samples"] = np.frombuffer(event_data_bytes[20:24], dtype=np.uint32)
        self.metadata_event["waveform"] = np.frombuffer(event_data_bytes[24:], dtype=np.uint16)
        return self._assemble_data_row()

    def _assemble_data_row(self):
        timestamp = self.metadata_event["timestamp"]
        energy = self.metadata_event["energy"]
        energy_short = self.metadata_event["energy_short"]
        flags = self.metadata_event["flags"]
        waveform = self.metadata_event["waveform"]
        df_event = pd.DataFrame([np.concatenate((timestamp, energy, energy_short, flags, waveform))])
        return df_event.rename(index=str, columns={0: "TIMETAG", 1: "E_LONG", 2: "E_SHORT", 3: "FLAGS"})

    def parse_xml(self, xmlfile):
        pass
