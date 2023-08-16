# module triplet
'''
Computes spectral lines for triplet oxygen.
'''

import matplotlib.pyplot as plt
from scipy.special import wofz # pylint: disable=no-name-in-module
import scienceplots # pylint: disable=unused-import
import pandas as pd
import numpy as np

import constants as cn

def line_initializer(k_qn: np.ndarray) -> np.ndarray:
    '''
    Creates an array of valid spectral lines by applying selection rules to the rotational quantum
    number K.

    Args:
        k_qn (np.ndarray): 1-d array of rotational quantum numbers

    Returns:
        np.ndarray: 1-d array of SpectralLine objects with populated attributes
    '''

    # Empty list to contain all valid spectral lines
    lines = []

    for i in k_qn:
        for j in k_qn:
            # Remove every even R and P line since the nuclear spin of oxygen is zero
            if (j % 2) != 0:
                # Selection rules for the R branch
                if k_qn[i] - k_qn[j] == 1:
                    main_branch = 'r'
                    satt_branch = 'qr'
                    for sb_1 in range(3):
                        lines.append(SpectralLine(i, j, main_branch, [sb_1, sb_1]))
                        for sb_2 in range(3):
                            if (sb_1 != sb_2) & (sb_1 > sb_2):
                                lines.append(SpectralLine(i, j, satt_branch, [sb_1, sb_2]))
                # Selection rules for the P branch
                elif k_qn[i] - k_qn[j] == -1:
                    main_branch = 'p'
                    satt_branch = 'qp'
                    for sb_1 in range(3):
                        lines.append(SpectralLine(i, j, main_branch, [sb_1, sb_1]))
                        for sb_2 in range(3):
                            if (sb_1 != sb_2) & (sb_1 < sb_2):
                                lines.append(SpectralLine(i, j, satt_branch, [sb_1, sb_2]))

    return np.array(lines)

def band_origin(states: list['State']) -> float:
    '''
    Calculates the 'band origin' for each state, which is the combination of vibrational and
    electrical energies.

    Args:
        states (list[State]): list of 2 State objects, the ground and excited states

    Returns:
        float: sum of electrical and vibrational energies
    '''
    ground_state = states[0]
    excite_state = states[1]

    elc_en = excite_state.electronic_term() - ground_state.electronic_term()
    vib_en = excite_state.vibrational_term() - ground_state.vibrational_term()

    return elc_en + vib_en

def rotational_term(k_qn: int, state: 'State', sub_branch: int) -> float:
    '''
    Calculates the rotational term F_i(K) for a given rotational quantum number.

    Args:
        k_qn (int): rotational quantum number
        state (State): the electronic state of the molecule
        sub_branch (int): either 1, 2, or 3

    Returns:
        float: F_i(K) depending on the sub branch of the spectral line
    '''

    first_term = state.rotational_terms()[0] * k_qn * (k_qn + 1) - \
                 state.rotational_terms()[1] * k_qn**2 * (k_qn + 1)**2 + \
                 state.rotational_terms()[2] * k_qn**3 * (k_qn + 1)**3

    # See footnote 2 on pg. 223 of Herzberg
    # For K = 1, the sign in front of the square root must be inverted
    if k_qn == 1:
        sqrt_sign = -1
    else:
        sqrt_sign = 1

    # TODO: reminder that the sign in front of state.spn_const[0] was changed from
    #       a - to a + on 8/7/23
    if sub_branch == 1:
        return first_term + (2 * k_qn + 3) * state.rotational_terms()[0] + \
               state.spn_const[0] - sqrt_sign * np.sqrt((2 * k_qn + 3)**2 * \
               state.rotational_terms()[0]**2 + state.spn_const[0]**2 - 2 * \
               state.spn_const[0] * state.rotational_terms()[0]) + \
               state.spn_const[1] * (k_qn + 1)

    if sub_branch == 2:
        return first_term

    return first_term - (2 * k_qn - 1) * state.rotational_terms()[0] - \
           state.spn_const[0] + sqrt_sign * np.sqrt((2 * k_qn - 1)**2 * \
           state.rotational_terms()[0]**2 + state.spn_const[0]**2 - 2 * \
           state.spn_const[0] * state.rotational_terms()[0]) - \
           state.spn_const[1] * k_qn

def normalization(ins: list) -> np.ndarray:
    '''
    Normalizes intensity data for plotting against experimental reference.

    Args:
        ins (list): list of intensity data

    Returns:
        np.ndarray: normalized data
    '''

    max_val = max(ins[0])

    return np.array(ins) / max_val

def convolve(v_c: float, v_0: float, temp: float, pres: float) -> float:
    '''
    Convolves spectral data using natural, doppler, and collisional broadening.

    Args:
        v_c (float): current continuous wavenumber
        v_0 (float): wavenumber peak
        temp (float): temperature of the gas
        pres (float): pressure of the gas

    Returns:
        float: intensity data at given wavenumber
    '''

    # Mass of molecular oxygen [kg]
    m_o2 = (2 * 15.999) / cn.AVOGD / 1e3
    # Collisional cross section of O2 with O2 (ground state radius) [cm]
    sigma_ab = np.pi * (cn.X_RAD + cn.X_RAD)**2
    # Reduced mass [kg]
    mu_ab = (m_o2 * m_o2) / (m_o2 + m_o2)

    # Natural FWHM [1/cm]
    gamma_n = sigma_ab**2 * np.sqrt(8 / (np.pi * mu_ab * cn.BOLTZ * temp)) / 4
    # Doppler FWHM [1/cm]
    sigma_v = v_0 * np.sqrt((cn.BOLTZ * temp) / (m_o2 * (cn.LIGHT / 1e2)**2))
    # Collision FWHM [1/cm]
    # Convert pressure in N/m^2 to pressure in dyne/cm^2
    gamma_v = (pres * 10) * sigma_ab**2 * np.sqrt(8 / (np.pi * mu_ab * cn.BOLTZ * temp)) / 2

    gamma = np.sqrt(gamma_n**2 + gamma_v**2)

    print(f'gaussian (natural + doppler): {gamma}')
    print(f'lorentzian (collisional): {sigma_v}')

    # Faddeeva function
    fadd = ((v_c - v_0) + 1j * gamma) / (sigma_v * np.sqrt(2))

    return np.real(wofz(fadd)) / (sigma_v * np.sqrt(2 * np.pi))

def sample_plotter() -> tuple:
    '''
    Reads sample files and converts them into plottable data.

    Returns:
        tuple: wavenumber and intensity data from each sample
    '''

    sample_data = []
    sample_data.append(pd.read_csv('../data/harvrd.csv', delimiter=' '))
    sample_data.append(pd.read_csv('../data/hitran.csv', delimiter=' '))
    sample_data.append(pd.read_csv('../data/pgopher.csv', delimiter=' '))
    sample_data.append(pd.read_csv('../data/webplot_09_band.csv', delimiter=' '))

    df = sample_data[3]
    # add band origin
    df['wavenumber'] = df['wavenumber'].add(36185)

    samp_wn = []
    samp_in = []
    for _, val in enumerate(sample_data):
        samp_wn.append(val['wavenumber'])
        samp_in.append(val['intensity'])

    for i, val in enumerate(samp_in):
        samp_in[i] = val / val.max()

    return samp_wn, samp_in

def plotter(line_data: list, conv_data: tuple, samp_data: tuple):
    '''
    Plots all available data.

    Args:
        line_data (list): discrete spectral line data calculated in the program
        conv_data (list): convolved continuous data calculated in the program
        samp_data (tuple): any sample data generated from .csv files
    '''

    mydpi = 96

    plt.figure(figsize=(1920/mydpi, 1080/mydpi), dpi=mydpi)

    plt.style.use(['science', 'grid'])
    #plt.stem(line_data[0][0], line_data[1][0], 'k', markerfmt='', label='R Branch')
    #plt.stem(line_data[0][1], line_data[1][1], 'r', markerfmt='', label='P Branch')
    #plt.stem(line_data[0][2], line_data[1][2], 'k', markerfmt='', label='QR Branch')
    #plt.stem(line_data[0][3], line_data[1][3], 'r', markerfmt='', label='QP Branch')
    plt.plot(conv_data[0], conv_data[1], label='Convolved Data')
    plt.plot(samp_data[0][0], samp_data[1][0], label='Harvard Data')
    #plt.stem(samp_data[0][1], samp_data[1][1], 'y', markerfmt='', label='HITRAN Data')
    #plt.stem(samp_data[0][2], samp_data[1][2], 'blue', markerfmt='', label='PGOPHER Data')
    #plt.plot(samp_data[0][3], samp_data[1][3], 'orange', label='Cosby 1993')
    #plt.xlim([min(samp_data[0][3]), max(samp_data[0][3])])
    plt.xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    #plt.savefig('../img/example.webp', dpi=mydpi * 2)
    plt.show()

def convolved_data(total_wn: list, total_in: list, temp: float, pres: float) -> tuple:
    '''
    Generates plottable convolved data using the convolution function.

    Args:
        total_wn (list): all wavenumbers from all branches
        total_in (list): all intensities from all branches
        temp (float): temperature of the gas
        pres (float): pressure of the gas

    Returns:
        tuple: convolved wavenumbers and intensities
    '''

    # Generate a fine-grained x-axis for plotting
    conv_wn = np.linspace(min(total_wn), max(total_wn), 10000)
    conv_in = np.zeros_like(conv_wn)

    # Convolve wavenumber peaks with chosen probability density function
    for wavepeak, i in zip(total_wn, total_in):
        conv_in += i * convolve(conv_wn, wavepeak, temp, pres)
    conv_in = conv_in / max(conv_in)

    return conv_wn, conv_in

class State:
    '''
    Object for storing values having to do with the electronic and vibrational state of a molecule.
    '''

    def __init__(self, constants: list, v_qn: int) -> None:
        self.elc_const = constants[0]
        self.vib_const = constants[1:5]
        self.rot_const = constants[5:12]
        self.spn_const = constants[12:14]
        self.v_qn      = v_qn

    def rotational_terms(self) -> list[float]:
        '''
        Calculates the rotational constants B_v, D_v, and H_v. [1/cm]

        Returns:
            list[float]: gives [b_v, d_v, h_v]
        '''

        b_v = self.rot_const[0] - \
              self.rot_const[1] * (self.v_qn + 0.5) + \
              self.rot_const[2] * (self.v_qn + 0.5)**2 + \
              self.rot_const[3] * (self.v_qn + 0.5)**3

        d_v = self.rot_const[4] - self.rot_const[5] * (self.v_qn + 0.5)

        h_v = self.rot_const[6]

        return [b_v, d_v, h_v]

    def electronic_term(self) -> float:
        '''
        Calculates the electronic term T_e. [1/cm]

        Returns:
            float: gives T_e
        '''

        return self.elc_const

    def vibrational_term(self) -> float:
        '''
        Calculates the vibrational term G(v) for a given vibrational quantum number. [1/cm]

        Returns:
            float: gives G(v)
        '''

        return self.vib_const[0] * (self.v_qn + 0.5) - \
               self.vib_const[1] * (self.v_qn + 0.5)**2 + \
               self.vib_const[2] * (self.v_qn + 0.5)**3 + \
               self.vib_const[3] * (self.v_qn + 0.5)**4

class SpectralLine:
    '''
    Object for containing valid spectral lines.
    '''

    def __init__(self, k_1: int, k_2: int, main_branch: str, sub_branch: list[int]) -> None:
        self.k_1         = k_1
        self.k_2         = k_2
        self.main_branch = main_branch
        self.sub_branch  = sub_branch

    def wavenumber(self, v_0: float, states: list['State']) -> float:
        '''
        Calculates the wavenumber of light emitted.

        Args:
            v_0 (float): band origin
            states (list[State]): a list of the ground and excited states

        Returns:
            float: wavenumber
        '''

        ground_state = states[0]
        excite_state = states[1]

        return v_0 + rotational_term(self.k_1, excite_state, self.sub_branch[0]) - \
                     rotational_term(self.k_2, ground_state, self.sub_branch[1])

    def intensity(self, v_0: float, states: list['State'], temp: float) -> float:
        '''
        Calculates the intensity of the light emitted.

        Args:
            v_0 (float): band origin
            states (list[State]]): list of ground and excited states
            temp (float): temperature of the gas

        Returns:
            float: intensity of light for a given line
        '''

        ground_state = states[0]
        part = (cn.BOLTZ * temp) / (cn.PLANC * cn.LIGHT * cn.X_BE)
        base = (self.wavenumber(v_0, states) / part) * \
               np.exp(- (rotational_term(self.k_2, ground_state, \
               self.sub_branch[0]) * cn.PLANC * cn.LIGHT) / (cn.BOLTZ * temp))

        if self.main_branch == 'r':
            linestr = ((self.k_2 + 1)**2 - 0.25) / (self.k_2 + 1)
            intn =  base * linestr
        elif self.main_branch == 'p':
            linestr  = ((self.k_2)**2 - 0.25) / (self.k_2)
            intn =  base * linestr
        else:
            linestr = (2 * self.k_2 + 1) / (4 * self.k_2 * (self.k_2 + 1))
            intn = base * linestr

        # Naive approach of applying 1:2:1 line intensity ratio to each band, this way the two peaks
        # on either side of the main peak have 1/2 the intensity

        # Note: this *seems* to be what PGOPHER is doing from what I can tell, also haven't been
        # able to find anything in Herzberg about it yet
        if self.sub_branch[0] == 0 or self.sub_branch[0] == 1:
            return intn / 2

        return intn

def main():
    '''
    Runs the program.
    '''

    # Temperature in Kelvin
    temp = 300
    # Pressure in Pa
    pres = 101325
    # Range of desired rotational quantum numbers
    k_qn = np.arange(0, 35, 1)

    # Instantiate the ground and excited states
    ground_state = State(cn.X_CONSTS, 0)
    excite_state = State(cn.B_CONSTS, 2)
    states = [ground_state, excite_state]

    # Calculate the band origin energy
    v_0 = band_origin(states)
    #v_0 = 36185

    # Initialize the list of valid spectral lines
    lines = line_initializer(k_qn)

    # Get the wavenumbers and intensities for each main branch (for plotting separately)
    branch_map = {'r': 0, 'p': 1, 'qr': 2, 'qp': 3}
    wns = [[line.wavenumber(v_0, states) for line in lines if line.main_branch == branch]
           for branch in branch_map]
    ins = [[line.intensity(v_0, states, temp) for line in lines if line.main_branch == branch]
           for branch in branch_map]
    ins = normalization(ins)

    # Combine the wavenumber and intensity data for use with the convolution function
    total_wn = [line.wavenumber(v_0, states) for line in lines]
    total_in = [line.intensity(v_0, states, temp) for line in lines]
    total_in = total_in / max(total_in)

    # Convolve the data
    conv_data = convolved_data(total_wn, total_in, temp, pres)

    # Fetch sample data for plotting
    smp_plot = sample_plotter()

    plotter([wns, ins], conv_data, smp_plot)

if __name__ == '__main__':
    main()
