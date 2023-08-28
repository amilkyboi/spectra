# module initialize
'''
Initializes each spectral line by applying valid selection rules.
'''

import numpy as np

from energy import State
import constants as cn
import energy

def selection_rules(rot_qn_list: np.ndarray, pd_data) -> np.ndarray:
    '''
    Initializes spectral lines with ground and excited state rotational quantum numbers, along with
    their respective branch index given the valid selection rules for triplet oxygen, i.e. ΔN = ±1.

    Args:
        rot_qn_list (np.ndarray): a list of rotational quantum numbers that are to be considered

    Returns:
        np.ndarray: array of SpectralLine objects
    '''

    # Empty list to contain all valid spectral lines
    lines = []

    for grnd_rot_qn in rot_qn_list:
        for exct_rot_qn in rot_qn_list:
            # Remove every even R and P line since the nuclear spin of oxygen is zero
            if grnd_rot_qn % 2 == 1:

                # Selection rules for the R branch
                if exct_rot_qn - grnd_rot_qn == 1:
                    for grnd_branch_idx in range(1, 4):
                        for exct_branch_idx in range(1, 4):
                            if grnd_branch_idx == exct_branch_idx:
                                lines.append(SpectralLine(grnd_rot_qn, exct_rot_qn, 'r',
                                                          grnd_branch_idx, exct_branch_idx, 0.0))
                            if grnd_branch_idx > exct_branch_idx:
                                lines.append(SpectralLine(grnd_rot_qn, exct_rot_qn, 'rq',
                                                          grnd_branch_idx, exct_branch_idx, 0.0))

                # Selection rules for the P branch
                elif exct_rot_qn - grnd_rot_qn == -1:
                    for grnd_branch_idx in range(1, 4):
                        for exct_branch_idx in range(1, 4):
                            if grnd_branch_idx == exct_branch_idx:
                                lines.append(SpectralLine(grnd_rot_qn, exct_rot_qn, 'p',
                                                          grnd_branch_idx, exct_branch_idx, 0.0))
                            if grnd_branch_idx < exct_branch_idx:
                                lines.append(SpectralLine(grnd_rot_qn, exct_rot_qn, 'pq',
                                                          grnd_branch_idx, exct_branch_idx, 0.0))

    for line in lines:
        if line.grnd_branch_idx == 1:
            line.predissociation = pd_data['f1'][pd_data['rot_qn'] == line.exct_rot_qn].iloc[0]
        elif line.grnd_branch_idx == 2:
            line.predissociation = pd_data['f2'][pd_data['rot_qn'] == line.exct_rot_qn].iloc[0]
        else:
            line.predissociation = pd_data['f3'][pd_data['rot_qn'] == line.exct_rot_qn].iloc[0]

    return np.array(lines)

class SpectralLine:
    '''
    Holds the necessary data for a single spectral line.
    '''

    def __init__(self, grnd_rot_qn: int, exct_rot_qn: int, branch: str, grnd_branch_idx: int,
                 exct_branch_idx: int, predissociation: float) -> None:
        self.grnd_rot_qn     = grnd_rot_qn
        self.exct_rot_qn     = exct_rot_qn
        self.branch          = branch
        self.grnd_branch_idx = grnd_branch_idx
        self.exct_branch_idx = exct_branch_idx
        self.predissociation = predissociation

    def wavenumber(self, band_origin: float, grnd_state: 'State', exct_state: 'State') -> float:
        '''
        Given the electronic, vibrational, and rotational term values, caluclates the wavenumnber
        (energy) of the resulting emission/absorption.

        Args:
            band_origin (float): electronic + vibrational term values
            grnd_state (State): ground state
            exct_state (State): excited state

        Returns:
            float: emitted/absorbed wavenumber
        '''

        return band_origin + \
               energy.rotational_term(self.exct_rot_qn, exct_state, self.exct_branch_idx) - \
               energy.rotational_term(self.grnd_rot_qn, grnd_state, self.grnd_branch_idx)

    def intensity(self, band_origin: float, grnd_state: 'State', exct_state: 'State',
                  temp: float) -> float:
        '''
        Uses the Gaussian distribution function to calculate the population density (and therefore
        intensity) of each spectral line.

        Args:
            band_origin (float): electronic + vibrational term values
            grnd_state (State): ground state
            exct_state (State): excited state
            temp (float): temperature

        Returns:
            float: intensity
        '''

        # Q_r, the total temperature-dependent partition function for the ground state
        part = (cn.BOLTZ * temp) / (cn.PLANC * cn.LIGHT * cn.X_BE)

        # The basic intensity function if no branches are considered
        base = (self.wavenumber(band_origin, grnd_state, exct_state) / part) * \
               np.exp(- (energy.rotational_term(self.grnd_rot_qn, grnd_state, \
               self.grnd_branch_idx) * cn.PLANC * cn.LIGHT) / (cn.BOLTZ * temp))

        # Intensity is dependent upon branch, with satellite branches having a much lower intensity
        # (notice that r and p scale with N**2, while rq and rp scale with 1/N**2)
        if self.branch == 'r':
            linestr = ((self.grnd_rot_qn + 1)**2 - 0.25) / (self.grnd_rot_qn + 1)
            intn =  base * linestr
        elif self.branch == 'p':
            linestr  = ((self.grnd_rot_qn)**2 - 0.25) / (self.grnd_rot_qn)
            intn =  base * linestr
        else:
            linestr = (2 * self.grnd_rot_qn + 1) / (4 * self.grnd_rot_qn * (self.exct_rot_qn + 1))
            intn = base * linestr

        # Naive approach of applying 1:2:1 line intensity ratio to each band, this way the two peaks
        # on either side of the main peak have 1/2 the intensity

        # NOTE: this *seems* to be what PGOPHER is doing from what I can tell, also haven't been
        #       able to find anything in Herzberg about it yet
        if self.grnd_branch_idx in (1, 3):
            return intn / 2

        return intn
