# module energy
'''
Used for calculating the energy term values for each state.
'''

import numpy as np

def get_band_origin(grnd_state: 'State', exct_state: 'State') -> float:
    '''
    Computes the electronic + vibrational term values.

    Args:
        grnd_state (State): ground state
        exct_state (State): excited state

    Returns:
        float: electronic + vibrational term values
    '''

    elc_energy = exct_state.electronic_term() - grnd_state.electronic_term()
    vib_energy = exct_state.vibrational_term() - grnd_state.vibrational_term()

    return elc_energy + vib_energy

def rotational_term(rot_qn: int, state: 'State', branch_idx: int) -> float:
    '''
    Calculates the rotational term value F(N).

    Args:
        rot_qn (int): current rotational quantum number
        state (State): current state
        branch_idx (int): branch index: r, p, or a satellite variation

    Returns:
        float: rotational term F(N)
    '''

    first_term = state.rotational_constants()[0] * rot_qn * (rot_qn + 1) - \
                 state.rotational_constants()[1] * rot_qn**2 * (rot_qn + 1)**2 + \
                 state.rotational_constants()[2] * rot_qn**3 * (rot_qn + 1)**3

    # See footnote 2 on pg. 223 of Herzberg
    # For N = 1, the sign in front of the square root must be inverted
    if rot_qn == 1:
        sqrt_sign = -1
    else:
        sqrt_sign = 1

    # NOTE: reminder that the sign in front of state.spn_const[0] was changed from a - to a + on
    #       8/7/23. I believe that the formula stated in Herzberg is wrong, as this change makes the
    #       results match experimental and simulated data more accurately.
    if branch_idx == 1:
        return first_term + (2 * rot_qn + 3) * state.rotational_constants()[0] + \
               state.spn_const[0] - sqrt_sign * np.sqrt((2 * rot_qn + 3)**2 * \
               state.rotational_constants()[0]**2 + state.spn_const[0]**2 - 2 * \
               state.spn_const[0] * state.rotational_constants()[0]) + \
               state.spn_const[1] * (rot_qn + 1)

    if branch_idx == 2:
        return first_term

    return first_term - (2 * rot_qn - 1) * state.rotational_constants()[0] - \
           state.spn_const[0] + sqrt_sign * np.sqrt((2 * rot_qn - 1)**2 * \
           state.rotational_constants()[0]**2 + state.spn_const[0]**2 - 2 * \
           state.spn_const[0] * state.rotational_constants()[0]) - \
           state.spn_const[1] * rot_qn

class State:
    '''
    Stores the necessary constants for either the ground state or the excited state of the current
    electronic transition.
    '''

    def __init__(self, constants: list, vib_qn: int) -> None:
        self.elc_const = constants[0]
        self.vib_const = constants[1:5]
        self.rot_const = constants[5:12]
        self.spn_const = constants[12:14]
        self.vib_qn    = vib_qn

    def rotational_constants(self) -> list[float]:
        '''
        Calculates the rotational constants B_v, D_v, and H_v.

        Returns:
            list[float]: [B_v, D_v, H_v]
        '''

        b_v = self.rot_const[0]                          - \
              self.rot_const[1] * (self.vib_qn + 0.5)    + \
              self.rot_const[2] * (self.vib_qn + 0.5)**2 + \
              self.rot_const[3] * (self.vib_qn + 0.5)**3

        d_v = self.rot_const[4] - self.rot_const[5] * (self.vib_qn + 0.5)

        h_v = self.rot_const[6]

        return [b_v, d_v, h_v]

    def electronic_term(self) -> float:
        '''
        Calculates the electronic term T_e.

        Returns:
            float: electronic term T_e
        '''

        return self.elc_const

    def vibrational_term(self) -> float:
        '''
        Calculates the vibrational term G(v).

        Returns:
            float: vibrational term G(v)
        '''
        return self.vib_const[0] * (self.vib_qn + 0.5)    - \
               self.vib_const[1] * (self.vib_qn + 0.5)**2 + \
               self.vib_const[2] * (self.vib_qn + 0.5)**3 + \
               self.vib_const[3] * (self.vib_qn + 0.5)**4
