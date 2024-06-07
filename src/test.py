# module test
"""
Contains temporary functions used for testing.
"""

import numpy as np
import pandas as pd
from scipy.special import wofz # pylint: disable=no-name-in-module

import constants as cn
from simtype import SimType
from molecule import Molecule
from simulation import Simulation

def plot_samp(samp_file: str):
    sample_data = pd.read_csv(f'../data/samples/{samp_file}.csv')

    wavenumbers = sample_data['wavenumbers'].to_numpy()
    intensities = sample_data['intensities'].to_numpy()
    intensities /= intensities.max()

    return wavenumbers, intensities

def convolve_brod(sim: Simulation, wavenumbers_line: np.ndarray, intensities_line: np.ndarray,
                  wavenumbers_conv: np.ndarray) -> np.ndarray:
    intensities_conv = np.zeros_like(wavenumbers_conv)
    natural, collide = broadening_params(sim)

    for _, (wave, intn) in enumerate(zip(wavenumbers_line, intensities_line)):
        intensities_conv += intn * broadening_fn(sim, wavenumbers_conv, wave, natural, collide)

    return intensities_conv

def broadening_fn(sim: Simulation, convolved_wavenumbers: np.ndarray, wavenumber_peak: float,
                  natural: float, collide: float) -> np.ndarray:

    doppler = (wavenumber_peak *
               np.sqrt(cn.BOLTZ * sim.temp / (sim.molecule.molecular_mass * (cn.LIGHT / 1e2)**2)))

    prediss = 0.1

    gauss = doppler
    loren = natural + collide + prediss

    fadd = ((convolved_wavenumbers - wavenumber_peak) + 1j * loren) / (gauss * np.sqrt(2))

    return np.real(wofz(fadd)) / (gauss * np.sqrt(2 * np.pi))

def broadening_params(sim: Simulation) -> tuple[float, float]:
    natural = (sim.state_lo.cross_section**2 *
               np.sqrt(8 / (np.pi * sim.molecule.reduced_mass * cn.BOLTZ * sim.temp)) / 4)

    collide = ((sim.pres * 10) * sim.state_lo.cross_section**2 *
               np.sqrt(8 / (np.pi * sim.molecule.reduced_mass * cn.BOLTZ * sim.temp)) / 2)

    return natural, collide

temp: float = 300.0
pres: float = 101325.0

mol_o2 = Molecule('o2', 'o', 'o')

bands_sim = [(2, 0)]

o2_sim = Simulation(mol_o2, temp, pres, np.arange(0, 36), 'b3su', 'x3sg', bands_sim,
                    SimType.ABSORPTION)

wns, ins = plot_samp('pgopher')

cwns = np.linspace(wns.min(), wns.max(), 10000)
cins = convolve_brod(o2_sim, wns, ins, cwns)
cins /= cins.max()

def wavenum_to_wavelen(x):
    x             = np.array(x, float)
    near_zero     = np.isclose(x, 0)
    x[near_zero]  = np.inf
    x[~near_zero] = 1 / x[~near_zero]

    return x * 1e7

cwls = wavenum_to_wavelen(cwns)
