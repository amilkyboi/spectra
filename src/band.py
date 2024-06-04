# module band

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

import terms
import convolve
import input as inp
from line import Line

if TYPE_CHECKING:
    from simulation import Simulation

class Band:
    def __init__(self, vib_qn_up: int, vib_qn_lo: int, sim: Simulation) -> None:
        self.vib_qn_up:     int        = vib_qn_up
        self.vib_qn_lo:     int        = vib_qn_lo
        self.sim:           Simulation = sim
        self.band_origin:   float      = self.get_band_origin()
        self.lines:         np.ndarray = self.get_allowed_lines()
        self.franck_condon: float      = self.sim.molecule.fc_data[self.vib_qn_up][self.vib_qn_lo]

    def get_lif_lines(self, rot_qn_up: int, rot_qn_lo: int) -> np.ndarray:
        # TODO: 06/04/24 - maybe make a separate class for LIF to bypass having to pass rot_qn_up
        #       and rot_qn_lo in here

        # NOTE: 06/04/24 - since the rotational partition function requires all the lines to be
        #       simulated to get an accurate result, the total number of lines in each band is
        #       always calculated

        lines = []

        if rot_qn_lo % 2:
            lines.extend(self.get_allowed_branches(rot_qn_up, rot_qn_lo))

        return np.array(lines)

    def get_allowed_lines(self) -> np.ndarray:
        lines = []

        for rot_qn_up in self.sim.rot_lvls:
            for rot_qn_lo in self.sim.rot_lvls:
                # for molecular oxygen, all transitions with even values of J'' are forbidden
                if rot_qn_lo % 2:
                    lines.extend(self.get_allowed_branches(rot_qn_up, rot_qn_lo))

        return np.array(lines)

    def get_allowed_branches(self, rot_qn_up: int, rot_qn_lo: int) -> list[float]:
        # determines the selection rules for Hund's case (b)
        # ∆N = ±1, ∆N = 0 is forbidden for Σ-Σ transitions
        # Herzberg p. 244, eq. (V, 44)

        lines = []

        # account for triplet splitting in the 3Σ-3Σ transition
        branch_range = range(1, 4)

        delta_rot_qn = rot_qn_up - rot_qn_lo

        # R branch
        if delta_rot_qn == 1:
            lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'r', 'rq'))

        # P branch
        elif delta_rot_qn == -1:
            lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'p', 'pq'))

        return lines

    def get_branch_idx(self, rot_qn_up: int, rot_qn_lo: int, branch_range: range, branch_main: str,
                       branch_secondary: str) -> list[float]:
        # determines the lines included in the transition
        # Herzberg pp. 249-251, eqs. (V, 48-53)

        lines = []

        for branch_idx_up in branch_range:
            for branch_idx_lo in branch_range:
                # main branches
                # R1, R2, R3, P1, P2, P3
                if branch_idx_up == branch_idx_lo:
                    lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up, branch_idx_lo,
                                      branch_main, self.sim, self, self.sim.molecule))
                # satellite branches
                # RQ31, RQ32, RQ21
                if (branch_idx_up > branch_idx_lo) & (branch_secondary == 'rq'):
                    lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up, branch_idx_lo,
                                      branch_secondary, self.sim, self, self.sim.molecule))
                # PQ13, PQ23, PQ12
                elif (branch_idx_up < branch_idx_lo) & (branch_secondary == 'pq'):
                    lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up, branch_idx_lo,
                                      branch_secondary, self.sim, self, self.sim.molecule))

        return lines

    def rotational_partition(self) -> float:
        # calculates the rotational partition function
        # Herzberg p. 125, eq. (III, 164)

        q_r = 0

        for line in self.lines:
            # NOTE: 05/07/24 - the Boltzmann factor and line strengths already change for emission
            #       versus absorption, so this function can remain as-is

            honl_london = line.honl_london_factor()
            boltzmann   = line.boltzmann_factor()

            q_r += honl_london * boltzmann

        return q_r

    def get_band_origin(self) -> float:
        # calculates the band origin
        # Herzberg p. 151, eq. (IV, 12)

        elc_energy = self.sim.state_up.consts['t_e'] - self.sim.state_lo.consts['t_e']

        vib_energy = (terms.vibrational_term(self.sim.state_up, self.vib_qn_up) -
                      terms.vibrational_term(self.sim.state_lo, self.vib_qn_lo))

        return elc_energy + vib_energy

    def wavenumbers_lif(self, rot_qn_up: int, rot_qn_lo: int) -> np.ndarray:
        return np.array([line.wavenumber() for line in self.get_lif_lines(rot_qn_up, rot_qn_lo)])

    def intensities_lif(self, rot_qn_up: int, rot_qn_lo: int) -> np.ndarray:
        intensities_lif = np.array([line.intensity() for line in
                                    self.get_lif_lines(rot_qn_up, rot_qn_lo)])

        intensities_lif /= intensities_lif.max()
        intensities_lif *= self.franck_condon / self.sim.max_fc

        return intensities_lif

    def wavenumbers_line(self) -> np.ndarray:
        return np.array([line.wavenumber() for line in self.lines])

    def intensities_line(self) -> np.ndarray:
        intensities_line = np.array([line.intensity() for line in self.lines])

        intensities_line /= intensities_line.max()
        intensities_line *= self.franck_condon / self.sim.max_fc

        return intensities_line

    def wavenumbers_conv(self) -> np.ndarray:
        wavenumbers_line = self.wavenumbers_line()

        return np.linspace(wavenumbers_line.min(), wavenumbers_line.max(), inp.GRANULARITY)

    def intensities_conv(self) -> np.ndarray:
        intensities_conv = convolve.convolve_brod(self.sim, self.lines, self.wavenumbers_line(),
                                                  self.intensities_line(), self.wavenumbers_conv())

        intensities_conv /= intensities_conv.max()
        intensities_conv *= self.franck_condon / self.sim.max_fc

        return intensities_conv

    def intensities_inst(self, broadening: float) -> np.ndarray:
        intensities_inst = convolve.convolve_inst(self.wavenumbers_conv(), self.intensities_conv(),
                                                  broadening)

        intensities_inst /= intensities_inst.max()
        intensities_inst *= self.franck_condon / self.sim.max_fc

        return intensities_inst
