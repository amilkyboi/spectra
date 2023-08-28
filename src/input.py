# module input
'''
User-defined control inputs for quantum numbers and plotting, among others.
'''

import numpy as np

# Global temperature and pressure
# By default, all vibrational transitions are considered at the same temperature and pressure
TEMP: float = 300.0
PRES: float = 101325.0

# Temperature used in Cosby (0, 9) is 300 K, pressure is 2666.45 Pa

# Rotational levels
# Cosby predissociation data only goes up to N = 36
ROT_LVLS: np.ndarray = np.arange(0, 37, 1)

# List of vibrational transitions considered in (v', v'') format
VIB_BANDS: list[tuple] = [(2, 0)]

# Band origin override
# Constants don't line up exactly for comparison with Cosby (0, 9) data, so the band origin can be
# set manually to get a better comparison
BAND_ORIG: tuple[bool, int] = (False, 36185)

# Line data
LINE_DATA: bool = False

# Convolved data
CONV_DATA: bool = True
# Granulatity of the convolved data
CONV_GRAN: int  = 10000

# Sample data
SAMP_DATA: bool = True
SAMP_FILE: list[str] = ['harvard']
SAMP_COLS: list[str] = ['purple']
SAMP_LABL: list[str] = ['Harvard Data']

# General plotting
PLOT_SAVE:  bool  = True
PLOT_PATH:  str   = '../img/example.webp'
DPI:        int   = 96
SCREEN_RES: tuple = (1920, 1080)

# Custom plot limits
SET_LIMS: tuple[bool, tuple] = (False, (36170, 36192))
# Custom font size
FONT_SIZE: tuple[bool, int] = (False, 20)
