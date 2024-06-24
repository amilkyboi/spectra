# module plot
"""
Contains functions used for plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # pylint: disable = unused-import

import convolve
from line import Line
from simulation import Simulation

# plt.style.use(["science", "grid"])

def wavenum_to_wavelen(x) -> np.ndarray:
    """
    Converts wavenumbers to wavelengths and vice versa.
    """

    x                     = np.array(x, float)
    near_zero: np.ndarray = np.isclose(x, 0)

    x[near_zero]  = np.inf
    x[~near_zero] = 1 / x[~near_zero]

    return x * 1e7

def plot_show() -> None:
    """
    Sets axis labels, creates a secondary x-axis for wavenumbers, displays the legend, and calls the
    plot.
    """

    ax = plt.gca()

    secax = ax.secondary_xaxis("top", functions=(wavenum_to_wavelen, wavenum_to_wavelen))
    secax.set_xlabel("Wavenumber, $\\nu$ [cm$^{-1}$]")

    plt.xlabel("Wavelength, $\\lambda$ [nm]")
    plt.ylabel("Intensity, Arbitrary Units [-]")

    plt.legend()
    plt.show()

def plot_samp(samp_file: str, color: str, plot_as: str = "stem") -> None:
    """
    Plots either line data or convolved data from a designated sample file.
    """

    sample_data: pd.DataFrame = pd.read_csv(f"../data/samples/{samp_file}.csv")

    wavenumbers: np.ndarray = sample_data["wavenumbers"].to_numpy()
    wavelengths: np.ndarray = wavenum_to_wavelen(wavenumbers)
    intensities: np.ndarray = sample_data["intensities"].to_numpy()
    intensities /= intensities.max()

    match plot_as:
        case "stem":
            plt.stem(wavelengths, intensities, color, markerfmt='', label=samp_file)
        case "plot":
            plt.plot(wavelengths, intensities, color, label=samp_file)
        case _:
            raise ValueError(f"Invalid value for plot_as: {plot_as}.")

def plot_line_info(sim: Simulation) -> None:
    """
    Plots information about each rotational line.
    """

    for vib_band in sim.vib_bands:
        wavenumbers_line: np.ndarray = vib_band.wavenumbers_line()
        wavelengths_line: np.ndarray = wavenum_to_wavelen(wavenumbers_line)
        intensities_line: np.ndarray = vib_band.intensities_line()
        lines:            list[Line] = vib_band.lines

        for idx, line in enumerate(lines):
            plt.text(wavelengths_line[idx], intensities_line[idx], f"{line.branch_name}")

def plot_line(sim: Simulation, colors: list) -> None:
    """
    Plots each rotational line.
    """

    for idx, vib_band in enumerate(sim.vib_bands):
        wavelengths_line: np.ndarray = wavenum_to_wavelen(vib_band.wavenumbers_line())

        plt.stem(wavelengths_line, vib_band.intensities_line(), colors[idx], markerfmt='',
                 label=f"{sim.molecule.name} {vib_band.vib_qn_up, vib_band.vib_qn_lo} line")

def plot_conv(sim: Simulation, colors: list) -> None:
    """
    Plots convolved data for each vibrational band separately.
    """

    for idx, vib_band in enumerate(sim.vib_bands):
        wavelengths_conv: np.ndarray = wavenum_to_wavelen(vib_band.wavenumbers_conv())

        # FIXME: 06/05/24 - Temporary normalization for rotational lines in a single band, used for
        #        comparing against sample data
        intensities_conv: np.ndarray = vib_band.intensities_conv()
        intensities_conv /= intensities_conv.max()

        plt.plot(wavelengths_conv, intensities_conv, colors[idx],
                 label=f"{sim.molecule.name} {vib_band.vib_qn_up, vib_band.vib_qn_lo} conv")

def plot_conv_all(sim: Simulation, color: str) -> None:
    """
    Plots convolved data for all vibrational bands simultaneously.
    """

    wavenumbers_conv, intensities_conv = sim.all_convolved_data()
    wavelengths_conv: np.ndarray = wavenum_to_wavelen(wavenumbers_conv)

    intensities_conv /= intensities_conv.max()

    plt.plot(wavelengths_conv, intensities_conv, color, label=f"{sim.molecule.name} conv all")

def plot_inst(sim: Simulation, colors: list, broadening: float) -> None:
    """
    Plots data convolved with an instrument function for each vibrational band separately.
    """

    for idx, vib_band in enumerate(sim.vib_bands):
        wavelengths_conv: np.ndarray = wavenum_to_wavelen(vib_band.wavenumbers_conv())

        plt.plot(wavelengths_conv, vib_band.intensities_inst(broadening), colors[idx],
                 label=f"{sim.molecule.name} {vib_band.vib_qn_up, vib_band.vib_qn_lo} inst")

def plot_inst_all(sim: Simulation, color: str, broadening: float) -> None:
    """
    Plots data convolved with an instrument function for all vibrational bands simultaneously.
    """

    wavenumbers_conv, intensities_conv = sim.all_convolved_data()
    wavelengths_conv: np.ndarray = wavenum_to_wavelen(wavenumbers_conv)

    intensities_inst: np.ndarray = convolve.convolve_inst(wavenumbers_conv, intensities_conv,
                                                          broadening)
    intensities_inst /= intensities_inst.max()

    plt.plot(wavelengths_conv, intensities_inst, color, label=f"{sim.molecule.name} inst all")

def plot_residual(sim: Simulation, color: str, samp_file: str) -> None:
    """
    Plots the difference between convolved simulation data and sample data.
    """

    # Sample processing
    sample_data: pd.DataFrame = pd.read_csv(f"../data/samples/{samp_file}.csv")

    wavenumbers_samp: np.ndarray = sample_data["wavenumbers"].to_numpy()
    intensities_samp: np.ndarray = sample_data["intensities"].to_numpy()
    intensities_samp /= intensities_samp.max()

    for _, vib_band in enumerate(sim.vib_bands):
        # FIXME: 06/05/24 - Temporary normalization for rotational lines in a single band, used for
        #        comparing against sample data
        wavenumbers_sim: np.ndarray = vib_band.wavenumbers_conv()
        intensities_sim: np.ndarray = vib_band.intensities_conv()
        intensities_sim /= intensities_sim.max()

        # Experimental data is held as the baseline, simulated data is linearly interpolated
        intensities_interp: np.ndarray = np.interp(wavenumbers_samp, wavenumbers_sim,
                                                   intensities_sim)

        residual:     np.ndarray = intensities_samp - intensities_interp
        abs_residual: np.ndarray = np.abs(residual)

        print(f"Max absolute residual: {abs_residual.max()}")
        print(f"Mean absolute residual: {abs_residual.mean()}")
        print(f"Standard deviation: {residual.std()}")

        plt.plot(wavenum_to_wavelen(wavenumbers_samp), residual, color,
                 label=f"{sim.molecule.name} {vib_band.vib_qn_up, vib_band.vib_qn_lo} residual")
