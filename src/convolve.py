# module convolve
"""
Contains functions used for convolution.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import wofz # pylint: disable=no-name-in-module

from line import Line
import constants as cn

if TYPE_CHECKING:
    from simulation import Simulation

def convolve_inst(wavenumbers_conv: np.ndarray, intensities_conv: np.ndarray,
                  broadening: float) -> np.ndarray:
    """
    Convolves a discrete number of spectral lines into a continuous spectra by applying an
    instrument function.
    """

    intensities_inst: np.ndarray = np.zeros_like(wavenumbers_conv)

    for wave, intn in zip(wavenumbers_conv, intensities_conv):
        intensities_inst += intn * instrument_fn(wavenumbers_conv, wave, broadening)

    return intensities_inst

def convolve_brod(sim: Simulation, lines: list[Line], wavenumbers_line: np.ndarray,
                  intensities_line: np.ndarray, wavenumbers_conv: np.ndarray) -> np.ndarray:
    """
    Convolves a discrete number of spectral lines into a continuous spectra by applying a broadening
    function.
    """

    intensities_conv: np.ndarray = np.zeros_like(wavenumbers_conv)
    natural, collide = broadening_params(sim)

    for idx, (wave, intn) in enumerate(zip(wavenumbers_line, intensities_line)):
        intensities_conv += intn * broadening_fn(sim, lines, wavenumbers_conv, wave, idx, natural,
                                                 collide)

    return intensities_conv

def instrument_fn(convolved_wavenumbers: np.ndarray, wavenumber_peak: float,
                  broadening: float) -> np.ndarray:
    """
    Simulates the effects of instrument broadening using a Gaussian probability density function.
    """

    return (np.exp(-0.5 * (convolved_wavenumbers - wavenumber_peak)**2 / broadening**2) /
            (broadening * np.sqrt(2 * np.pi)))

def broadening_fn(sim: Simulation, lines: list[Line], convolved_wavenumbers: np.ndarray,
                  wavenumber_peak: float, line_idx: int, natural: float,
                  collide: float) -> np.ndarray:
    """
    Simulates the effects of collisional, Doppler, natural, and predissociation broadening using a
    Voigt probability density function.
    """

    # Doppler broadening: [1/cm]
    # Princeton Quantitative Laser Diagnostics p. 13
    # Converts speed of light in [cm/s] to [m/s]
    doppler: float = (wavenumber_peak * np.sqrt(cn.BOLTZ * sim.temp /
                      (sim.molecule.molecular_mass * (cn.LIGHT / 1e2)**2)))

    # Predissociation broadening: [1/cm]
    prediss: float = lines[line_idx].predissociation()

    gauss: float = doppler
    loren: float = natural + collide + prediss

    # Faddeeva function
    fadd: np.ndarray = (((convolved_wavenumbers - wavenumber_peak) + 1j * loren) /
                        (gauss * np.sqrt(2)))

    return np.real(wofz(fadd)) / (gauss * np.sqrt(2 * np.pi))

def broadening_params(sim: Simulation) -> tuple[float, float]:
    """
    Computes the broadening parameters that do not depend on wavelength or individual lines.
    """

    # FIXME: 06/06/24 - Not sure where the source for this came from, so I'm removing it for now by
    #        setting it equal to zero; the magnitude is around 1e-8, which is mostly negligible
    # Natural broadening: [1/cm]
    natural: float = 0.0
    # natural = (sim.state_lo.cross_section**2 *
    #            np.sqrt(8 / (np.pi * sim.molecule.reduced_mass * cn.BOLTZ * sim.temp)) / 4)

    # Collisional broadening: [1/cm]
    # Princeton Quantitative Laser Diagnostics p. 10
    # Converts pressure in [N/m^2] to [dyne/cm^2]
    collide: float = ((sim.pres * 10) * sim.state_lo.cross_section**2 *
                      np.sqrt(8 / (np.pi * sim.molecule.reduced_mass * cn.BOLTZ * sim.temp)) / 2)

    return natural, collide
