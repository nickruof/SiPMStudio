import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sys.path.append("/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/")
from SiPMStudio.core import data_loading
from SiPMStudio.core import devices
from SiPMStudio.core import digitizers

from SiPMStudio.processing import processor
from SiPMStudio.processing import process_data

path = "/Users/nickruof/Documents/LEGEND/SiPM/ketek_data/waves_31/UNFILTERED/0_waves_31.csv"
path2 = "/Users/nickruof/Documents/LEGEND/SiPM/ketek_data/waves_32/UNFILTERED/0_waves_32.csv"
Digitizer = digitizers.CAENDT5730(df_data=path)
Digitizer.v_range = 2.0
Digitizer.e_cal = 2.0e-15
proc = processor.Processor()
proc.set_processor(Digitizer)
proc.add(fun_name="adc_to_volts", settings={"digitizer":Digitizer})
proc.add(fun_name="baseline_subtract", settings={})
proc.add(fun_name="moving_average", settings={"box_size":19})
proc.add(fun_name="savgol", settings={"window":27, "order":2})

output_directory = "/Users/nickruof/Documents/LEGEND/SiPM/software_drafts/SiPMStudio/tests/"
process_data.ProcessData(data_files=[path, path2], processor=proc, output_dir=output_directory, multiprocess=False, digitizer=Digitizer, chunk=2000)


