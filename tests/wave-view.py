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
from SiPMStudio.analysis import dark
from SiPMStudio.plots import plots
from SiPMStudio.processing.functions import fit_multi_gauss

path = "/Users/nickruof/Documents/LEGEND/SiPM/ketek_data/waves_32/UNFILTERED/0_waves_32.csv"
Digitizer = digitizers.CAENDT5730(df_data=path)
Digitizer.v_range = 2.0
proc = processor.Processor()
proc.set_processor(Digitizer)
proc.add(fun_name="adc_to_volts", settings={"digitizer":Digitizer})
proc.add(fun_name="baseline_subtract", settings={})
#proc.add(fun_name="wavelet_denoise", settings={"wavelet":"db1", "levels":4, "mode":"soft"})
proc.add(fun_name="savgol", settings={"window":27, "order":2})
result = proc.process()
for i in range(0, 20):
    plots.plot_waveform(proc.waves.iloc[i, :], find_peaks=True, min_height=0.002, min_dist=70)