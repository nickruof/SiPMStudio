import numpy as np
import scipy.constants as const

from scipy.optimize import curve_fit
from SiPMStudio.processing.functions import gaussian


def get_mode(n, bin_centers, values):
    max_value = max(n)
    max_loc = list(n).index(max(n))
    coeffs, covs = curve_fit(gaussian, bin_centers, n, p0=[max_value, bin_centers[max_loc], np.std(values)])
    errors = np.sqrt(np.diag(covs))
    return coeffs[1], errors[1]
                 

def apd_photons(apd_charge, photon_wavelength, response, area):
    photon_energy = const.h * const.c / photon_wavelength[0]
    num_photons = apd_charge[0] / (photon_energy * response * area)
    error = (1/(const.h*const.c*response*area))*np.sqrt(apd_charge[0]**2*photon_wavelength[1]**2 + photon_wavelength[0]**2*apd_charge[1]**2)
    return num_photons, error


def sipm_photons(sipm_triggers, cross_talk, ap, area):
    num_photons = sipm_triggers[0] / (1 + cross_talk)/(1 + ap) / area
    error1 = sipm_triggers[1] / (1 + cross_talk) / (1 + ap) / area
    error2 = sipm_triggers[0]*cross_talk[1] / ((1 + cross_talk[0])**2*(1 + ap[0])*area)
    error3 = sipm_triggers[0]*ap[1] /((1 + cross_talk[0])*(1 + ap[0])**2*area)
    error = np.sqrt(error1**2 + error2**2 + error3**2)
    return num_photons, error


def pde_calc(sipm_sig, apd_sig):
    pde = sipm_sig[0] / apd_sig[0]
    error = np.sqrt((sipm_sig[1]/apd_sig[0])**2 + (sipm_sig[0]*apd_sig[1]/apd_sig[0]**2)**2)
    return pde, error
