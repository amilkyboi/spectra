# module input
'''
User-defined control inputs for quantum numbers and plotting, among others.
'''

import numpy as np

# Global temperature and pressure
# By default, all vibrational transitions are considered at the same temperature and pressure
TEMP: float = 300.0
PRES: float = 2666.45

# Temperature used in Cosby (0, 9) is 300 K, pressure is 2666.45 Pa

# Rotational levels
# Cosby predissociation data only goes up to N = 36
ROT_LVLS: np.ndarray = np.arange(0, 37, 1)

# List of vibrational transitions considered in (v', v'') format
VIB_BANDS: list[tuple] = [(0, 9)]

# Band origin override
# Constants don't line up exactly for comparison with Cosby (0, 9) data, so the band origin can be
# set manually to get a better comparison
BAND_ORIG: tuple[bool, int] = (True, 36185)

# Line data
LINE_DATA: bool = True

# Convolved data
CONV_DATA: bool = True
# Granulatity of the convolved data
CONV_GRAN: int  = 10000

# Sample data
SAMP_DATA: bool = True
SAMP_FILE: list[str] = ['cosby09']
SAMP_COLS: list[str] = ['purple']
SAMP_LABL: list[str] = ['Cosby Data']

# General plotting
PLOT_SAVE:  bool  = False
PLOT_PATH:  str   = '../img/example.webp'
DPI:        int   = 96
SCREEN_RES: tuple = (1920, 1080)

# Custom plot limits
SET_LIMS: tuple[bool, tuple] = (True, (36170, 36192))
