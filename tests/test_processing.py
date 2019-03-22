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
from SiPMStudio.processing.functions import fit_multi_gauss

from scipy.stats import expon

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
#plots.plot_waveform(proc.waves.iloc[20, :], find_peaks=True, min_height=0.002, min_dist=70)

#Dark Analysis
ketek_32 = devices.sipm(name="ketek", area=9)
ketek_32.bias = [32]
bins = np.linspace(start=0, stop=max(Digitizer2.format_data()["E_SHORT"]), num=int(max(Digitizer2.format_data()["E_SHORT"])))
bin_vals, _bin_edges = np.histogram(Digitizer2.format_data()["E_SHORT"], bins=bins)
params = fit_multi_gauss(bins=bins, bin_vals=bin_vals, min_height=5000, min_dist=30)
#plots.pc_spectrum(hist_array=Digitizer2.format_data()["E_SHORT"], params=params)
dark.gain(params=params, digitizer=Digitizer, sipm=ketek_32)
dark.cross_talk(params_data=proc.calcs, params=params, sipm=ketek_32)
(dts, heights) = dark.delay_time_vs_height(params_data=proc.calcs, wave_data=proc.waves, min_height=0.002, min_dist=50)
dark.dcr_exp_fit(dts=dts, sipm=ketek_32, bounds=[0, 1e5])
dark.pulse_rate(wave_data=proc.waves, sipm=ketek_32, min_height=0.0015, min_dist=50)
#plots.plot_delay_height(dts, heights, density=True)
#plots.plot_delay_times(dts, fit=True)

print(params)
print(ketek_32.gain_magnitude)
print(ketek_32.cross_talk)
print(ketek_32.dcr_fit)
print(ketek_32.pulse_rate)
print(light.excess_charge_factor(sipm=ketek_32))

#Light Analysis
pde_path = "/Users/nickruof/Documents/LEGEND/SIPM/PDE/ketek_PDE/"
dark_file = "dark32.csv"
light_file = "lit32.csv"
scope = data_loading.Keithley2450()
leak = light.average_leakage(dataloader=scope, sipm=ketek_32, bias=[32], files=[pde_path+dark_file])
ref_diode = devices.photodiode("ref_diode", area=9)
ref_diode.load_response(pde_path+"responsivity.csv")
green_led = devices.led("green", wavelength=5.75e-7)
dark_files = [pde_path+"diodedark.csv", pde_path+dark_file]
light_files = [pde_path+"diodelit.csv", pde_path+light_file]
pde = light.continuous_pde(dataloader=scope, sipm=ketek_32, diode=ref_diode, led=green_led, bias=[32], dark_files=dark_files, light_files=light_files)

print(leak)
print(pde)
