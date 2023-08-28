# module main
'''
Computes spectral lines for triplet oxygen. See the README for details on implementation along with
available features.
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import output as out
import input as inp
import bands

def main():
    '''
    Runs the program.
    '''

    # Read in the table of Franck-Condon factors
    fc_data = np.loadtxt('../data/harris_rkr_fc.csv', delimiter=' ')

    # Read in the table of predissociation constants from Cosby
    pd_data = pd.read_csv('../data/predissociation.csv', delimiter=' ')

    # Create a vibrational band line plot for each of the user-selected bands
    band_list = []
    for band in inp.VIB_BANDS:
        band_list.append(bands.LinePlot(inp.TEMP, inp.PRES, inp.ROT_LVLS, band))

    # Find the maximum Franck-Condon factor of all the bands, this is used to normalize the
    # intensities of each band with respect to the largest band
    max_fc = max((band.get_fc(fc_data) for band in band_list))

    out.plot_style()

    # Since each set of lines is plotted separately, matplotlib doesn't know to cycle colors after
    # each one is plotted

    # This generates a list of hex values that are fed to the plot generators
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Automatic coloring and labeling is done for both line and convolved plots
    if inp.LINE_DATA:
        line_data   = [band.get_line(fc_data, max_fc, pd_data) for band in band_list]
        line_colors = color_list[0:len(line_data)]
        line_labels = [str(band) + ' Band' for band in inp.VIB_BANDS]

        out.plot_line(line_data, line_colors, line_labels)

    if inp.CONV_DATA:
        conv_data   = [band.get_conv(fc_data, max_fc, pd_data) for band in band_list]
        conv_colors = color_list[len(conv_data):2*len(conv_data)]
        conv_labels = ['Convolved ' + str(band) + ' Band' for band in inp.VIB_BANDS]

        out.plot_conv(conv_data, conv_colors, conv_labels)

    # Colors and labels for sample data are set in the input.py file
    if inp.SAMP_DATA:
        samp_data   = []
        for file in inp.SAMP_FILE:
            samp_data.append(out.configure_samples(file))
        out.plot_samp(samp_data, inp.SAMP_COLS, inp.SAMP_LABL)

    out.show_plot()

if __name__ == '__main__':
    main()
