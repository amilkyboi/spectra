# module constants
'''
Physical and diatomic constants for the X3Σg- to B3Σu- transition of oxygen.
'''

# Global Constants
## Physical Constants
BOLTZ = 1.380649e-23   # Boltzmann Constant [J/K]
PLANC = 6.62607015e-34 # Planck Constant    [J*s]
LIGHT = 2.99792458e10  # Speed of Light     [cm/s]
AVOGD = 6.02214076e23  # Avodagro Constant  [1/mol]

# Equilibrium values of the X 3Σg- and B 3Σu- states

## X3Σg-
X_TE     = 0.0         # minimum electronic energy                           [1/cm]

X_WE     = 1580.193    # vibrational constant - first term                   [1/cm]
X_WE_XE  = 11.981      # vibrational constant - second term                  [1/cm]
X_WE_YE  = 0.04747     # vibrational constant - third term                   [1/cm]
X_WE_ZE  = -0.001273   # vibrational constant - fourth term                  [1/cm]

X_BE     = 1.4456      # rotational constant in equilibrium position         [1/cm]

# Above value from Creek - A comprehensive re-analysis of the 02 Schumann-Runge band system - Tbl 5
X_B0 = 1.4376766 #- from NIST, note that no equilibrium value is published in NIST

X_ALPH_E = 0.01593     # rotational constant - first term                    [1/cm]
X_GAMM_E = 0.0         # rotation-vibration interaction constant             [1/cm]
X_DELT_E = 0.0         # rotational constant - third term                    [1/cm]

X_DE     = 4.839e-6    # centrifugal distortion constant                     [1/cm]
X_BETA_E = 0.0         # rotational constant - first term, centrifugal force [1/cm]

X_HE     = 0.0         # quartic centrifugal distortion constant             [1/cm]

X_RE     = 1.20752     # equilibium internuclear distance                    [Å]
X_V00    = 0.0         # position of 0-0 band

# These are for v = 0 according to PGOPHER
X_LAMD   = 1.9847511   # spin-spin constant                                  [1/cm]
X_GAMM   = -0.00842536 # spin-rotation constant                              [1/cm]

X_RAD    = 1.20752e-8  # internuclear distance                               [cm]

X_CONSTS = [X_TE, X_WE, X_WE_XE, X_WE_YE, X_WE_ZE, X_BE, X_ALPH_E, X_GAMM_E, X_DELT_E, X_DE, \
            X_BETA_E, X_HE, X_LAMD, X_GAMM]

## B3Σu-
B_TE     = 49793.28 # minimum electronic energy                           [1/cm]

B_WE     = 709.31   # vibrational constant - first term                   [1/cm]
B_WE_XE  = 10.65    # vibrational constant - second term                  [1/cm]
B_WE_YE  = 	-0.139  # vibrational constant - third term                   [1/cm]
B_WE_ZE  = 0.0      # vibrational constant - fourth term                  [1/cm]

B_BE     = 0.81902  # rotational constant in equilibrium position         [1/cm]
B_ALPH_E = 0.01206  # rotational constant - first term                    [1/cm]
B_GAMM_E = -5.56e-4 # rotation-vibration interaction constant             [1/cm]
B_DELT_E = 0.0      # rotational constant - third term                    [1/cm]

B_DE     = 4.55e-6  # centrifugal distortion constant                     [1/cm]
B_BETA_E = 0.22e-6  # rotational constant - first term, centrifugal force [1/cm] (v <= 4)

B_HE     = 0.0      # quartic centrifugal distortion constant             [1/cm]

B_RE     = 1.60426  # equilibium internuclear distance                    [Å]
B_V00    = 49358.15 # position of 0-0 band

B_LAMD   = 1.5      # spin-spin constant                                  [1/cm] (v <= 12)
B_GAMM   = 0.04     # spin-rotation constant                              [1/cm] (v <= 12)

B_RAD    = 1.60426e-8 # internuclear distance                             [cm]

B_CONSTS = [B_TE, B_WE, B_WE_XE, B_WE_YE, B_WE_ZE, B_BE, B_ALPH_E, B_GAMM_E, B_DELT_E, B_DE, \
            B_BETA_E, B_HE, B_LAMD, B_GAMM]
