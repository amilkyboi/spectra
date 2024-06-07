# module band
"""
Contains the implementation of the Band class.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

import terms
import convolve
import input as inp
from line import Line
import constants as cn
from state import State
from simtype import SimType

if TYPE_CHECKING:
    from simulation import Simulation

class Band:
    """
    A vibrational band containing multiple rotational lines.
    """

    def __init__(self, vib_qn_up: int, vib_qn_lo: int, sim: Simulation) -> None:
        self.vib_qn_up:     int        = vib_qn_up
        self.vib_qn_lo:     int        = vib_qn_lo
        self.sim:           Simulation = sim
        self.band_origin:   float      = self.get_band_origin()
        self.lines:         list[Line] = self.get_allowed_lines()
        self.franck_condon: float      = self.sim.molecule.fc_data[self.vib_qn_up][self.vib_qn_lo]
        self.rot_part:      float      = self.rotational_partition()
        self.vib_boltz:     float      = self.vib_boltzmann_factor()

    def get_allowed_lines(self) -> list[Line]:
        """
        Returns a list of allowed spectral lines.
        """

        lines: list[Line] = []

        for rot_qn_up in self.sim.rot_lvls:
            for rot_qn_lo in self.sim.rot_lvls:
                # For molecular oxygen, all transitions with even values of N'' are forbidden
                if rot_qn_lo % 2:
                    lines.extend(self.get_allowed_branches(rot_qn_up, rot_qn_lo))

        return lines

    def get_allowed_branches(self, rot_qn_up: int, rot_qn_lo: int) -> list[Line]:
        """
        Determines the selection rules for Hund's case (b).
        """

        # ∆N = ±1, ∆N = 0 is forbidden for Σ-Σ transitions
        # Herzberg p. 244, eq. (V, 44)

        lines: list[Line] = []

        # Account for triplet splitting in the 3Σ-3Σ transition
        branch_range: range = range(1, 4)

        delta_rot_qn: int = rot_qn_up - rot_qn_lo

        # R branch
        if delta_rot_qn == 1:
            lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'r', 'rq'))

        # P branch
        elif delta_rot_qn == -1:
            lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'p', 'pq'))

        return lines

    def get_branch_idx(self, rot_qn_up: int, rot_qn_lo: int, branch_range: range, branch_main: str,
                       branch_secondary: str) -> list[Line]:
        """
        Determines the lines included in the transition.
        """

        # Herzberg pp. 249-251, eqs. (V, 48-53)

        lines: list[Line] = []

        for branch_idx_up in branch_range:
            for branch_idx_lo in branch_range:
                # Main branches
                # R1, R2, R3, P1, P2, P3
                if branch_idx_up == branch_idx_lo:
                    lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up, branch_idx_lo,
                                      branch_main, self.sim, self, self.sim.molecule))
                # Satellite branches
                # RQ31, RQ32, RQ21
                if (branch_idx_up > branch_idx_lo) & (branch_secondary == 'rq'):
                    lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up, branch_idx_lo,
                                      branch_secondary, self.sim, self, self.sim.molecule))
                # PQ13, PQ23, PQ12
                elif (branch_idx_up < branch_idx_lo) & (branch_secondary == 'pq'):
                    lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up, branch_idx_lo,
                                      branch_secondary, self.sim, self, self.sim.molecule))

        return lines

    def vib_boltzmann_factor(self) -> float:
        """
        Returns the vibrational Boltzmann factor.
        """

        # Herzberg p. 123, eq. (III, 159)

        match self.sim.sim_type:
            case SimType.ABSORPTION:
                state:  State = self.sim.state_lo
                vib_qn: int   = self.vib_qn_lo
            case SimType.EMISSION | SimType.LIF:
                state:  State = self.sim.state_up
                vib_qn: int   = self.vib_qn_up
            case _:
                raise ValueError('Invalid SimType.')

        return np.exp(-terms.vibrational_term(state, vib_qn) * cn.PLANC * cn.LIGHT /
                      (cn.BOLTZ * self.sim.temp))

    def rotational_partition(self) -> float:
        """
        Returns the rotational partition function.
        """

        # Herzberg p. 125, eq. (III, 164)

        q_r: float = 0.0

        # NOTE: 06/05/24 - This *should* always include the maximum number of lines possible, i.e.
        #       the limit as the number of lines goes to infinity; the rotational quantum numbers go
        #       up to 35, so this should be quite accurate
        for line in self.lines:
            # NOTE: 05/07/24 - The Boltzmann factor and line strengths already change for emission
            #       versus absorption, so this function can remain as-is
            honl_london: float = line.honl_london_factor()
            boltzmann:   float = line.rot_boltzmann_factor()

            q_r += honl_london * boltzmann

        return q_r

    def get_band_origin(self) -> float:
        """
        Returns the band origin.
        """

        # Herzberg p. 151, eq. (IV, 12)

        elc_energy: float = self.sim.state_up.consts['t_e'] - self.sim.state_lo.consts['t_e']

        vib_energy: float = (terms.vibrational_term(self.sim.state_up, self.vib_qn_up) -
                             terms.vibrational_term(self.sim.state_lo, self.vib_qn_lo))

        return elc_energy + vib_energy

    def wavenumbers_line(self) -> np.ndarray:
        """
        Returns an array of wavenumbers for each line.
        """

        return np.array([line.wavenumber() for line in self.lines])

    def intensities_line(self) -> np.ndarray:
        """
        Returns an array of intensities for each line.
        """

        intensities_line: np.ndarray = np.array([line.intensity() for line in self.lines])

        # Normalize w.r.t. vibrational partition function
        intensities_line *= self.vib_boltz / self.sim.vib_part
        # Normalize w.r.t. Franck-Condon factor
        intensities_line *= self.franck_condon / self.sim.max_fc

        return intensities_line

    def wavenumbers_conv(self) -> np.ndarray:
        """
        Returns an array of convolved wavenumbers.
        """

        wavenumbers_line: np.ndarray = self.wavenumbers_line()

        # Generate a fine-grained x-axis using existing wavenumber data
        return np.linspace(wavenumbers_line.min(), wavenumbers_line.max(), inp.GRANULARITY)

    def intensities_conv(self) -> np.ndarray:
        """
        Returns an array of convolved intensities.
        """

        intensities_conv: np.ndarray = convolve.convolve_brod(self.sim, self.lines,
                                                              self.wavenumbers_line(),
                                                              self.intensities_line(),
                                                              self.wavenumbers_conv())

        intensities_conv *= self.vib_boltz / self.sim.vib_part
        intensities_conv *= self.franck_condon / self.sim.max_fc

        return intensities_conv

    def intensities_inst(self, broadening: float) -> np.ndarray:
        """
        Returns an array of convolved intensities.
        """

        intensities_inst: np.ndarray = convolve.convolve_inst(self.wavenumbers_conv(),
                                                              self.intensities_conv(),
                                                              broadening)

        intensities_inst *= self.vib_boltz / self.sim.vib_part
        intensities_inst *= self.franck_condon / self.sim.max_fc

        return intensities_inst
