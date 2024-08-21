# module simulation
"""
Contains the implementation of the Simulation class.
"""

import numpy as np

import terms
import convolve
import input as inp
from band import Band
from line import Line
import constants as cn
from state import State
from simtype import SimType
from molecule import Molecule

class Simulation:
    """
    A simulation containing multiple vibrational bands.
    """

    def __init__(self, molecule: Molecule, temp: float, pres: float, rot_lvls: np.ndarray,
                 state_up: str, state_lo: str, band_list: list[tuple[int, int]],
                 sim_type: SimType) -> None:
        self.molecule:  Molecule   = molecule
        self.temp:      float      = temp
        self.pres:      float      = pres
        self.rot_lvls:  np.ndarray = rot_lvls
        self.sim_type:  SimType    = sim_type
        self.state_up:  State      = State(state_up, self.molecule.consts)
        self.state_lo:  State      = State(state_lo, self.molecule.consts)
        self.vib_bands: list[Band] = [Band(vib_band[0], vib_band[1], self)
                                      for vib_band in band_list]
        self.max_fc:    float      = max(vib_band.franck_condon for vib_band in self.vib_bands)
        self.vib_part:  float      = self.vibrational_partition()

    def vibrational_partition(self) -> float:
        """
        Returns the vibrational partition function.
        """

        # Herzberg p. 123, eq. (III, 159)

        state: State

        match self.sim_type:
            case SimType.ABSORPTION:
                state = self.state_lo
            case SimType.EMISSION | SimType.LIF:
                state = self.state_up

        q_v: float = 0.0

        # NOTE: 06/05/24 - Since the partition function relies on the sum as the vibrational quantum
        #       number goes to infinity, the currently selected vibrational bands are not used;
        #       since the user might only have a single band selected, it would not correctly
        #       generate the state sum (the range from 0 to 18 is somewhat arbitrary though)
        for vib_qn in range(0, 19):
            q_v += np.exp(-terms.vibrational_term(state, vib_qn) * cn.PLANC * cn.LIGHT /
                          (cn.BOLTZ * self.temp))

        return q_v

    def all_line_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Combines the line data for all vibrational bands.
        """

        wavenumbers_line: np.ndarray = np.array([])
        intensities_line: np.ndarray = np.array([])

        for vib_band in self.vib_bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, vib_band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, vib_band.intensities_line()))

        return wavenumbers_line, intensities_line

    def all_conv_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates common axes for plotting the convolved data of all vibrational bands at once.
        """

        wavenumbers_line: np.ndarray = np.array([])
        intensities_line: np.ndarray = np.array([])
        lines:            list[Line] = []

        for vib_band in self.vib_bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, vib_band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, vib_band.intensities_line()))
            lines.extend(vib_band.lines)

        wavenumbers_conv = np.linspace(wavenumbers_line.min(), wavenumbers_line.max(),
                                       inp.GRANULARITY)
        intensities_conv = convolve.convolve_brod(self, lines, wavenumbers_line, intensities_line,
                                                  wavenumbers_conv)

        return wavenumbers_conv, intensities_conv
