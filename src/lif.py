# module lif
"""
Contains example spectra for LIF.
"""

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

import plot
from molecule import Molecule
from lif_utils import LifSimulation, plot_lif, plot_lif_info

def plot_show() -> None:
    ax = plt.gca()

    secax = ax.secondary_xaxis("top", functions=(plot.wavenum_to_wavelen, plot.wavenum_to_wavelen))
    secax.set_xlabel("Wavenumber, $\\nu$ [cm$^{-1}$]")

    plt.xlabel("Wavelength, $\\lambda$ [nm]")
    plt.ylabel("Intensity, Arbitrary Units [-]")

    plt.legend()
    plt.show()

def main():
    """
    Construct example spectra here.
    """

    temp: float = 300.0
    pres: float = 101325.0

    upper_band: int = 4
    lower_band: int = 2

    upper_lif: list[tuple[int, int]] = [(upper_band, v) for v in range(18, -1, -1)]
    lower_lif: list[tuple[int, int]] = [(lower_band, v) for v in range(18, -1, -1)]

    o2_mol: Molecule = Molecule("o2", 'o', 'o')

    o2_up: LifSimulation = LifSimulation(o2_mol, temp, pres, np.arange(0, 36), "b3su", "x3sg",
                                         upper_lif)

    o2_lo: LifSimulation = LifSimulation(o2_mol, temp, pres, np.arange(0, 36), "b3su", "x3sg",
                                         lower_lif)

    palette: list[tuple] = plt.cycler("color", plt.cm.tab20c.colors).by_key()["color"]
    colors:  list[str]   = [matplotlib.colors.to_hex(color) for color in palette]

    plot_lif(o2_up, 12, 13, colors)
    plot_lif(o2_lo, 16, 15, colors)
    plot_lif_info(o2_up, 12, 13)
    plot_lif_info(o2_lo, 16, 15)
    plot_show()

    # NOTE: 06/05/24 - The Franck-Condon factors of all 18 v'' bands sharing the same v' must to be
    #       considered since their intensities are normalized relative to the highest intensity in
    #       the given simulation

    # Only search the main triplet in non-satellite bands
    upper_wavenumbers: np.ndarray = np.array([])
    upper_intensities: np.ndarray = np.array([])
    upper_lines:       np.ndarray = np.array([])

    lower_wavenumbers: np.ndarray = np.array([])
    lower_intensities: np.ndarray = np.array([])
    lower_lines:       np.ndarray = np.array([])

    for vib_band in o2_up.vib_bands:
        upper_wavenumbers = np.concatenate((upper_wavenumbers, vib_band.wavenumbers_line()))
        upper_intensities = np.concatenate((upper_intensities, vib_band.intensities_line()))
        upper_lines       = np.concatenate((upper_lines, vib_band.lines))

    for vib_band in o2_lo.vib_bands:
        lower_wavenumbers = np.concatenate((lower_wavenumbers, vib_band.wavenumbers_line()))
        lower_intensities = np.concatenate((lower_intensities, vib_band.intensities_line()))
        lower_lines       = np.concatenate((lower_lines, vib_band.lines))

    # Now that all the intensities from every band are collected together, we have to normalize both
    # of them by the same factor, otherwise their relative intensities will not be preserved
    max_intensity: float = max(upper_intensities.max(), lower_intensities.max())
    upper_intensities /= max_intensity
    lower_intensities /= max_intensity

    # Filter by intensity (can't be done per line since the vibrational bands hold the information
    # about the normalized intensity of the lines)
    upper_indices = np.where(upper_intensities > 0.001)
    lower_indices = np.where(lower_intensities > 0.001)

    upper_wavenumbers = upper_wavenumbers[upper_indices]
    upper_intensities = upper_intensities[upper_indices]
    upper_lines       = upper_lines[upper_indices]

    lower_wavenumbers = lower_wavenumbers[lower_indices]
    lower_intensities = lower_intensities[lower_indices]
    lower_lines       = lower_lines[lower_indices]

    # Compare all wavenumbers against each other and find nearby lines
    diff_matrix = np.abs(upper_wavenumbers[:, np.newaxis] - lower_wavenumbers)
    pair_mask = diff_matrix < 1

    upper_indices, lower_indices = np.where(pair_mask)

    upper_wavenumbers = upper_wavenumbers[upper_indices]
    upper_intensities = upper_intensities[upper_indices]
    upper_lines       = upper_lines[upper_indices]

    lower_wavenumbers = lower_wavenumbers[lower_indices]
    lower_intensities = lower_intensities[lower_indices]
    lower_lines       = lower_lines[lower_indices]

    upper_wavelengths = plot.wavenum_to_wavelen(upper_wavenumbers)
    lower_wavelengths = plot.wavenum_to_wavelen(lower_wavenumbers)

    plt.stem(upper_wavelengths, upper_intensities, 'b', markerfmt='')
    for idx, line in enumerate(upper_lines):
        plt.text(upper_wavelengths[idx], upper_intensities[idx],
                 f"v: {line.band.vib_qn_up, line.band.vib_qn_lo}\n"
                 f"J: {int(line.rot_qn_up), int(line.rot_qn_lo)}")

    plt.stem(lower_wavelengths, lower_intensities, 'r', markerfmt='')
    for idx, line in enumerate(lower_lines):
        plt.text(lower_wavelengths[idx], lower_intensities[idx],
                 f"v: {line.band.vib_qn_up, line.band.vib_qn_lo}\n"
                 f"J: {int(line.rot_qn_up), int(line.rot_qn_lo)}")

    plt.show()

if __name__ == "__main__":
    main()
