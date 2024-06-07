# module test
"""
Contains example spectra for absorption and emission.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, to_hex

import plot
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

    bands: list[tuple[int, int]] = [(2, 0)] # for x in range(0, 7) for v in range(18, -1, -1)]

    o2_mol: Molecule = Molecule('o2', 'o', 'o')

    o2_sim: Simulation = Simulation(o2_mol, temp, pres, np.arange(0, 36), 'b3su', 'x3sg', bands,
                                    SimType.ABSORPTION)

    colors_small: list[str] = plt.rcParams['axes.prop_cycle'].by_key()['color']

    palette:    list[tuple] = plt.cycler('color', plt.cm.tab20c.colors).by_key()['color']
    colors_mid: list[str]   = [to_hex(color) for color in palette]

    cmap:         Colormap  = plt.get_cmap('rainbow')
    num_bands:    int       = len(bands)
    colors_large: list[str] = [to_hex(cmap(i / num_bands)) for i in range(num_bands)]

    # FIXME: 05/06/24 - Each time a plot is called (plot_line, plot_info, etc.), the vibrational
    #        bands are iterated through, meaning the wavelength and intensity info for each band is
    #        potentially being re-calculated several times

    # FIXME: 05/06/24 - The intensities are no longer normalized after adding the vibrational
    #        partition function; normalization would have to be performed on all vibrational bands
    #        at once

    plot.plot_conv(o2_sim, colors_small)
    plot.plot_samp('harvard/harvard20', colors_small[1], 'plot')
    plot.plot_residual(o2_sim, colors_small[2], 'harvard/harvard20')

    # Testing how the PGOPHER data compares when convolved (estimating predissociation rates)
    # from test import cwls, cins

    # plt.plot(cwls, cins, 'black', label='pgopher')

    plot.plot_show()

if __name__ == '__main__':
    main()
