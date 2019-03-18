import sys
sys.path.append("/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/")
from SiPMStudio.core import data_loading
from SiPMStudio.core import devices
from SiPMStudio.core import digitizers

from SiPMStudio.processing import processor
from SiPMStudio.analysis import dark
from SiPMStudio.plots import plots

path = "/Users/nickruof/Documents/LEGEND/SiPM/ketek_data_cold/waves_28/FILTERED/0_waves_28.csv"
Digitizer = digitizers.CAENDT5730(df_data=path)
Digitizer.v_range = 2.0
proc = processor.Processor()
proc.set_processor(Digitizer)
proc.add(fun_name="adc_to_volts", settings={"digitizer":Digitizer})
proc.add(fun_name="baseline_subtract", settings={})
proc.add(fun_name="savgol", settings={"window":15, "order":2})
result = proc.process()
plots.plot_waveform(proc.waves[0], find_peaks=True, min_height=0.002, min_dist=80)
#(dts, heights) = dark.delay_time_vs_height(proc.calcs, proc.waves, min_height=0.0003, min_dist=100)
