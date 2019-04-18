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
from SiPMStudio.analysis import light
from SiPMStudio.plots import plots

from sklearn.mixture import GaussianMixture

path = "/Users/nickruof/Documents/LEGEND/SiPM/ketek_data/waves_32/UNFILTERED/0_waves_32.csv"
path2 = "/Users/nickruof/Documents/LEGEND/SiPM/ketek_data/run_32/UNFILTERED/0_run_32.csv"
Digitizer = digitizers.CAENDT5730(df_data=path)
Digitizer2 = digitizers.CAENDT5730(df_data=path2)
Digitizer.v_range = 2.0
Digitizer.e_cal = 2.0e-15
proc = processor.Processor()
proc.set_processor(Digitizer)
proc.add(fun_name="adc_to_volts", settings={"digitizer":Digitizer})
proc.add(fun_name="baseline_subtract", settings={})
proc.add(fun_name="moving_average", settings={"box_size":19})
proc.add(fun_name="savgol", settings={"window":27, "order":2})
result = proc.process()

ketek_32 = devices.sipm(name="ketek", area=9e-6)
ketek_32.bias = [32]
bins = np.linspace(start=0, stop=max(Digitizer2.format_data()["E_SHORT"]), num=int(max(Digitizer2.format_data()["E_SHORT"])))
bin_vals, bin_edges = np.histogram(Digitizer2.format_data()["E_SHORT"], bins=bins)

dark.fit_multi_gauss(params_data=Digitizer2.format_data()["E_SHORT"], min_dist=80, min_height=1.0e-5, display=True)
