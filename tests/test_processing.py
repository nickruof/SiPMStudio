import sys
import numpy as np
sys.path.append("/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/")
from SiPMStudio.core import data_loading
from SiPMStudio.core import devices
from SiPMStudio.core import digitizers

from SiPMStudio.processing import processor
from SiPMStudio.analysis import dark
from SiPMStudio.plots import plots
from SiPMStudio.processing.functions import fit_multi_gauss

path = "/Users/nickruof/Documents/LEGEND/SiPM/ketek_data_cold/waves_28/FILTERED/0_waves_28.csv"
Digitizer = digitizers.CAENDT5730(df_data=path)
Digitizer.v_range = 2.0
proc = processor.Processor()
proc.set_processor(Digitizer)
proc.add(fun_name="adc_to_volts", settings={"digitizer":Digitizer})
proc.add(fun_name="baseline_subtract", settings={})
proc.add(fun_name="savgol", settings={"window":51, "order":3})
result = proc.process()
print(proc.calcs)
#plots.plot_waveform(proc.waves.iloc[10, :], find_peaks=True, min_height=0.002, min_dist=80)

#Dark Analysis
#bins = np.linspace(start=0, stop=max(result["E_short"]), num=int(max(proc.calcs["E_short"])))
#bin_vals, _bin_edges = np.histogram(proc.calcs["E_short"], bins=bins)
#params = fit_multi_gauss(bins=bins, bin_vals=bin_vals, min_height=100, min_dist=70)
#plots.pc_spectrum(hist_array=proc.calcs["E_short"])
