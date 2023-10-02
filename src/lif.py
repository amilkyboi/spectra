# module lif
'''
Testing LIF simulation.
'''

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import cm
import scienceplots # pylint: disable=unused-import
import pandas as pd
import numpy as np

import input as inp
import bands

plt.style.use(['science', 'grid'])
plt.figure(figsize=(inp.SCREEN_RES[0]/inp.DPI, inp.SCREEN_RES[1]/inp.DPI), dpi=inp.DPI)
plt.rcParams.update({'font.size': inp.FONT_SIZE[1]})

def main():
    '''
    Runs the program.
    '''

    # Read in the table of Franck-Condon factors
    fc_data = np.loadtxt('../data/franck-condon/cheung_rkr_fc.csv', delimiter=' ')

    # Read in the table of predissociation constants from Cosby
    pd_data = pd.read_csv('../data/predissociation.csv', delimiter=' ')

    # Excited and initial ground state vibrational quantum numbers
    excite_vib_qn = 7
    initial_ground_vib_qn = 0
    max_ground_vib_qn = 14

    band = bands.LinePlot(inp.TEMP, inp.PRES, inp.ROT_LVLS, (excite_vib_qn, initial_ground_vib_qn))
    max_fc = band.get_fc(fc_data)
    line = band.get_line(fc_data, max_fc, pd_data)

    plt.stem(line[0], line[1], 'k', markerfmt='', label='Total Band')
    plt.stem(line[0][116], line[1][116], 'r', markerfmt='', label='Selected Line')
    plt.title(f"Initial Excitation: $(v',v'') = ({excite_vib_qn}, {initial_ground_vib_qn})$, \
                Pressure: {inp.PRES} Pa, Temperature: {inp.TEMP} K")
    plt.xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.show()

    band_list = []
    labels    = []
    for grnd_vib_qn in range(initial_ground_vib_qn, max_ground_vib_qn + 1):
        band_list.append(bands.LinePlot(inp.TEMP, inp.PRES, inp.ROT_LVLS, (excite_vib_qn, grnd_vib_qn)))
        labels.append(grnd_vib_qn)

    max_fc = max((band.get_fc(fc_data) for band in band_list))

    line_data = [band.get_line(fc_data, max_fc, pd_data) for band in band_list]

    # Grab a rainbow colormap from the built-in matplotlib cmaps
    cmap = cm.get_cmap('rainbow')
    num_lines = len(line_data)

    # Assign each line a color, each being equally spaced within the colormap
    colors = [mcolors.to_hex(cmap(i / (num_lines - 1))) for i in range(num_lines)]

    fig, ax = plt.subplots(figsize=(inp.SCREEN_RES[0]/inp.DPI, inp.SCREEN_RES[1]/inp.DPI), dpi=inp.DPI)

    wavns = []
    intns = []
    for i, (wave, intn) in enumerate(line_data):
        wavns.append(wave[116])
        intns.append(intn[116])

    intns /= max(intns)

    for i, (wave, intn) in enumerate(zip(wavns, intns)):
        markerline, stemlines, baseline = ax.stem((1 / wave) * 1e7, intn, colors[i], markerfmt='',
                                                    label=f"$v''={labels[i]}$")
        plt.setp(stemlines, 'linewidth', 3)

    ax.set_title(f"Initial Excitation: $(v',v'') = ({excite_vib_qn}, {initial_ground_vib_qn})$, \
                   Pressure: {inp.PRES} Pa, Temperature: {inp.TEMP} K")
    ax.set_xlabel('Wavelength $\\nu$, [nm]')
    ax.set_ylabel('Normalized Intensity')
    ax.legend()

    # Convert from wavenumber to wavelength
    def wn2wl(wns):
        return (1 / wns) * 1e7

    # Add a secondary axis for wavelength
    secax = ax.secondary_xaxis('top', functions=(wn2wl, wn2wl))
    secax.set_xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    plt.show()

if __name__ == '__main__':
    main()
