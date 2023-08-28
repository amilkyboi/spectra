# module bands
'''
Holds the LinePlot class, which is different for each vibrational transition considered. Here, the
Franck-Condon factor is computed for each spectral plot and influences the final intensity data.
'''

import numpy as np

import convolution as conv
import initialize as init
import constants as cn
import input as inp
import energy

class LinePlot:
    '''
    Each LinePlot is a separate vibrational band.
    '''

    def __init__(self, temp: float, pres: float, rot_qn_list: np.ndarray,
                 states: tuple[int, int]) -> None:
        self.temp         = temp
        self.pres         = pres
        self.rot_qn_list  = rot_qn_list
        self.states       = states

    def get_fc(self, fc_data: np.ndarray) -> float:
        '''
        From the global Franck-Condon data array, grabs the correct FC factor for the current
        vibrational transition.

        Args:
            fc_data (np.ndarray): global Franck-Condon array

        Returns:
            float: Franck-Condon factor for the current vibrational transition
        '''

        return fc_data[self.states[0]][self.states[1]]

    def get_line(self, fc_data: np.ndarray, max_fc: float,
                 pd_data) -> tuple[np.ndarray, np.ndarray]:
        '''
        Finds the wavenumbers and intensities for each line in the plot.

        Args:
            fc_data (np.ndarray): global Franck-Condon array
            max_fc (float): maximum Franck-Condon factor from all vibrational transitions considered

        Returns:
            tuple[list, list]: (wavenumbers, intensities)
        '''

        # Initialize ground and excited states
        exct_state = energy.State(cn.B_CONSTS, self.states[0])
        grnd_state = energy.State(cn.X_CONSTS, self.states[1])

        # Calculate the band origin energy
        if inp.BAND_ORIG[0]:
            band_origin = inp.BAND_ORIG[1]
        else:
            band_origin = energy.get_band_origin(grnd_state, exct_state)

        # Initialize the list of valid spectral lines
        lines = init.selection_rules(self.rot_qn_list, pd_data)

        # Get the wavenumbers and intensities
        wns = np.array([line.wavenumber(band_origin, grnd_state, exct_state)
                        for line in lines])
        ins = np.array([line.intensity(band_origin, grnd_state, exct_state, self.temp)
                        for line in lines])

        # Normalize the intensity data w.r.t. the largest value
        # This is normalization of the plot with respect to itself
        ins /= ins.max()

        # Find the ratio between the largest Franck-Condon factor and the current plot
        norm_fc = self.get_fc(fc_data) / max_fc

        # This is normalization of the plot with respect to others
        ins *= norm_fc

        return wns, ins

    def get_conv(self, fc_data: np.ndarray, max_fc: float,
                 pd_data) -> tuple[np.ndarray, np.ndarray]:
        '''
        Finds the wavenumbers and intensities for the convolved data.

        Args:
            fc_data (np.ndarray): global Franck-Condon array
            max_fc (float): maximum Franck-Condon factor from all vibrational transitions considered

        Returns:
            tuple[np.ndarray, np.ndarray]: (wavenumbers, intensities)
        '''

        lines = init.selection_rules(self.rot_qn_list, pd_data)

        wns, ins = self.get_line(fc_data, max_fc, pd_data)

        conv_wns, conv_ins = conv.convolved_data(wns, ins, self.temp, self.pres, lines)

        conv_ins *= self.get_fc(fc_data) / max_fc

        return conv_wns, conv_ins
