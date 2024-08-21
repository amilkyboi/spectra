# module test
"""
Contains example spectra for absorption and emission.
"""

import numpy as np

import plot
from colors import get_colors
from molecule import Molecule
from simulation import Simulation, SimType

def main():
    """
    Construct example spectra here.
    """

    # NOTE: 05/06/24 - Only O2 is working right now; selection rules, etc. for other molecules are
    #       currently not implemented

    temp: float = 300.0
    pres: float = 101325.0

    bands: list[tuple[int, int]] = [(x, v) for x in range(0, 7) for v in range(18, -1, -1)]

    o2_mol: Molecule = Molecule("o2", 'o', 'o')

    o2_sim: Simulation = Simulation(o2_mol, temp, pres, np.arange(0, 36), "b3su", "x3sg", bands,
                                    SimType.ABSORPTION)

    # FIXME: 05/06/24 - Each time a plot is called (plot_line, plot_info, etc.), the vibrational
    #        bands are iterated through, meaning the wavelength and intensity info for each band is
    #        potentially being re-calculated several times

    colors: list[str] = get_colors("large", bands)

    plot.plot_line(o2_sim, colors)

    # plot.plot_conv(o2_sim, colors)
    # plot.plot_samp("harvard/harvard20", colors[1], "plot")
    # plot.plot_residual(o2_sim, colors[2], "harvard/harvard20")

    # Testing how the PGOPHER data compares when convolved (estimating predissociation rates)
    # from test import cwls, cins

    # plt.plot(cwls, cins, "black", label="pgopher")

    plot.plot_show()

if __name__ == "__main__":
    main()
