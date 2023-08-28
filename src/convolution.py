# module convolution
'''
Convolves the quantized spectral data by applying thermal doppler broadening, pressure broadening,
natural broadening, and predissociation broadening.
'''

from scipy.special import wofz # pylint: disable=no-name-in-module
import numpy as np

import constants as cn
import input as inp

def convolve(convolved_wavenumbers: np.ndarray, wavenumber_peak: float,
             temp: float, pres: float, idx, lines) -> float:
    '''
    Each step of the convolution involves the calculation of several broadening parameters, which
    depend on several factors. After these are calculated, this returns the Voigt probability
    distribution function.

    Args:
        convolved_wavenumbers (np.ndarray): array of all valid wavenumbers in the current plot
        wavenumber_peak (float): current quantized wavenumber
        temp (float): temperature
        pres (float): pressure

    Returns:
        float: Voigt probability distribution function
    '''

    # TODO: these calculations should be removed from the function considering that the mass and
    # such don't change for each iteration of the convolution.

    # Mass of molecular oxygen [kg]
    mass_o2 = (2 * 15.999) / cn.AVOGD / 1e3
    # Collisional cross section of O2 with O2 (ground state radius) [cm]
    cross_sec = np.pi * (cn.X_RAD + cn.X_RAD)**2
    # Reduced mass [kg]
    reduced_mass = (mass_o2 * mass_o2) / (mass_o2 + mass_o2)

    # Natural (Lorentzian)
    natural = cross_sec**2 * np.sqrt(8 / (np.pi * reduced_mass * cn.BOLTZ * temp)) / 4

    # Doppler (Gaussian)
    doppler = wavenumber_peak * np.sqrt((cn.BOLTZ * temp) / (mass_o2 * (cn.LIGHT / 1e2)**2))

    # Collision (Lorentzian)
    # Convert pressure in N/m^2 to pressure in dyne/cm^2
    collide = (pres * 10) * cross_sec**2 * np.sqrt(8 / (np.pi * reduced_mass * cn.BOLTZ * temp)) / 2

    # Predissociation (Lorentzian)
    prediss = lines[idx].predissociation

    # TODO: this might be wrong, not sure if the parameters just add together or what
    gauss = doppler
    loren = natural + collide + prediss

    # Faddeeva function
    fadd = ((convolved_wavenumbers - wavenumber_peak) + 1j * loren) / (gauss * np.sqrt(2))

    return np.real(wofz(fadd)) / (gauss * np.sqrt(2 * np.pi))

def convolved_data(wavenumbers: np.ndarray, intensities: np.ndarray,
                   temp: float, pres: float, lines) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates the final convolved data.

    Args:
        wavenumbers (np.ndarray): all quantized wavenumbers
        intensities (np.ndarray): matching intensities
        temp (float): temperature
        pres (float): pressure

    Returns:
        tuple[np.ndarray, np.ndarray]: convolved wavenumbers and intensities (both continuous)
    '''

    # Generate a fine-grained x-axis for plotting
    convolved_wavenumbers = np.linspace(wavenumbers.min(), wavenumbers.max(), inp.CONV_GRAN)
    convolved_intensities = np.zeros_like(convolved_wavenumbers)

    # Convolve wavenumber peaks with chosen probability density function
    for idx, (wavenumber_peak, intensity_peak) in enumerate(zip(wavenumbers, intensities)):
        convolved_intensities += intensity_peak * convolve(convolved_wavenumbers, wavenumber_peak,
                                                            temp, pres, idx, lines)
    convolved_intensities /= convolved_intensities.max()

    return convolved_wavenumbers, convolved_intensities
