# module term
"""
Contains functions used for vibrational and rotational term calculations.
"""

import numpy as np

from state import State

def vibrational_term(state: State, vib_qn: int) -> float:
    """
    Returns the vibrational term value.
    """

    # Herzberg p. 149, eq. (IV, 10)

    return (state.consts["w_e"]   * (vib_qn + 0.5)    -
            state.consts["we_xe"] * (vib_qn + 0.5)**2 +
            state.consts["we_ye"] * (vib_qn + 0.5)**3 +
            state.consts["we_ze"] * (vib_qn + 0.5)**4)

def rotational_constants(state: State, vib_qn: int) -> list[float]:
    """
    Returns the rotational constants.
    """

    # Herzberg pp. 107-109, eqs. (III, 117-127)

    b_v: float = (state.consts["b_e"]                        -
                  state.consts["alph_e"] * (vib_qn + 0.5)    +
                  state.consts["gamm_e"] * (vib_qn + 0.5)**2 +
                  state.consts["delt_e"] * (vib_qn + 0.5)**3)

    d_v: float = state.consts["d_e"] - state.consts["beta_e"] * (vib_qn + 0.5)

    h_v: float = state.consts["h_e"]

    return [b_v, d_v, h_v]

def rotational_term(state: State, vib_qn: int, rot_qn: int, branch_idx: int) -> float:
    """
    Returns the rotational term value.
    """

    b, d, h = rotational_constants(state, vib_qn)

    lamd: float = state.consts["lamd"]
    gamm: float = state.consts["gamm"]

    # Shorthand notation for rotational quantum numbers
    x1: int = (rot_qn + 1) * (rot_qn + 2) # F1: J = N + 1, so J(J + 1) -> (N + 1)(N + 2)
    x2: int = rot_qn * (rot_qn + 1)       # F2: J = N,     so J(J + 1) -> N(N + 1)
    x3: int = rot_qn * (rot_qn - 1)       # F3: J = N - 1, so J(J + 1) -> N(N - 1)

    # Bergeman, 1972 - The Fine Structure of O2
    # f1 = b*x1 + b - d*x1**2 - 6*d*x1 - 2*d - lamd/3 - gamm/2 - np.sqrt(16*b**2*x1 + 4*b**2 - 64*b*d*x1**2 - 80*b*d*x1 - 16*b*d - 8*b*lamd - 16*b*gamm*x1 - 4*b*gamm + 64*d**2*x1**3 + 144*d**2*x1**2 + 96*d**2*x1 + 16*d**2 + 16*d*lamd*x1 + 16*d*lamd + 32*d*gamm*x1**2 + 40*d*gamm*x1 + 8*d*gamm + 4*lamd**2 + 4*lamd*gamm + 4*gamm**2*x1 + gamm**2)/2
    # f2 = x2 * b - x2**2 * d + 2*lamd/3
    # f3 = b*x3 + b - d*x3**2 - 6*d*x3 - 2*d - lamd/3 - gamm/2 + np.sqrt(16*b**2*x3 + 4*b**2 - 64*b*d*x3**2 - 80*b*d*x3 - 16*b*d - 8*b*lamd - 16*b*gamm*x3 - 4*b*gamm + 64*d**2*x3**3 + 144*d**2*x3**2 + 96*d**2*x3 + 16*d**2 + 16*d*lamd*x3 + 16*d*lamd + 32*d*gamm*x3**2 + 40*d*gamm*x3 + 8*d*gamm + 4*lamd**2 + 4*lamd*gamm + 4*gamm**2*x3 + gamm**2)/2

    # Schlapp, 1936 - Fine Structure in the 3Σ Ground State of the Oxygen Molecule
    # From Herzberg - simplified expression for the square root)
    # f1 = b * rot_qn * (rot_qn + 1) + b * (2 * rot_qn + 3) - lamd - np.sqrt(b**2 * (2 * rot_qn + 3)**2 + lamd**2 - 2 * lamd * b) + gamm * (rot_qn + 1)
    # f2 = b * rot_qn * (rot_qn + 1)
    # f3 = b * rot_qn * (rot_qn + 1) - b * (2 * rot_qn - 1) - lamd + np.sqrt(b**2 * (2 * rot_qn - 1)**2 + lamd**2 - 2 * lamd * b) - gamm * rot_qn
    # For N = 1, J = 0 (F3 only), the sign in front of the square root has to be inverted

    # Schlapp, 1936 - Fine Structure in the 3Σ Ground State of the Oxygen Molecule
    # From matrix elements - "precise" values
    f1: float = b * x1 + b - lamd - np.sqrt((b - lamd)**2 + (b - gamm / 2)**2 * 4 * x1)
    f2: float = b * x2
    f3: float = b * x3 + b - lamd + np.sqrt((b - lamd)**2 + (b - gamm / 2)**2 * 4 * x3)

    # NOTE: 06/07/24 - The rotational quantum number being input into this function is N
    # TODO: 06/07/24 - When N = 0, only the F1 triplet exists for the ground state
    # Hanson - Spectroscopy and Optical Diagnostics for Gases p. 170

    # TODO: 06/07/24 - For J = 0, the energy is -2 * lamd + b * rot_qn * (rot_qn + 1) + 2 * b
    # Hougen - The Calculation of Rotational Energy Levels in Diatomic Molecules, p. 15
    # TODO: 06/07/24 -  J is only zero when N = 1 and the triplet branch is F3
    # Hanson - Spectroscopy and Optical Diagnostics for Gases, p. 170

    match branch_idx:
        # F1 triplet
        case 1:
            return f1
        # F2 triplet
        case 2:
            return f2
        # F3 triplet
        case 3:
            return f3
        case _:
            raise ValueError("ERROR: invalid branch index.")
