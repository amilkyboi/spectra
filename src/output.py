# module output
'''
Handles the plotting of line, convolved, and sample data. Also reads and configures sample data.
'''

import matplotlib.pyplot as plt
import scienceplots # pylint: disable=unused-import
import pandas as pd
import numpy as np

import input as inp

def plot_style() -> None:
    '''
    Sets a consistent plot style and output resolution based on the input screen parameters.
    '''

    plt.style.use(['science', 'grid'])
    plt.figure(figsize=(inp.SCREEN_RES[0]/inp.DPI, inp.SCREEN_RES[1]/inp.DPI), dpi=inp.DPI)

    if inp.FONT_SIZE[0]:
        plt.rcParams.update({'font.size': inp.FONT_SIZE[1]})

def show_plot():
    '''
    Sets global plot labels and saves the figure if necessary.
    '''

    if inp.SET_LIMS[0]:
        plt.xlim(inp.SET_LIMS[1][0], inp.SET_LIMS[1][1])

    plt.title(f'{inp.PRES} Pa, {inp.TEMP} K')
    plt.xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    plt.ylabel('Normalized Intensity')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Convert from wavenumber to wavelength
    def wn2wl(wns):
        return (1 / wns) * 1e7

    ax = plt.gca()

    # Add a secondary axis for wavelength
    secax = ax.secondary_xaxis('top', functions=(wn2wl, wn2wl))
    secax.set_xlabel('Wavelength $\\lambda$, [nm]')

    if inp.PLOT_SAVE:
        plt.savefig(inp.PLOT_PATH, dpi=inp.DPI * 2)
    else:
        plt.show()

def configure_samples(samp_file: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Reads the selected sample file into a pandas dataframe, returns the wavenumbers and intensities
    of the data.

    Args:
        samp_file (str): name of the sample file

    Returns:
        tuple[np.ndarray, np.ndarray]: wavenumber and intensity data
    '''

    sample_data = pd.read_csv(f'../data/{samp_file}.csv', delimiter=' ')

    if samp_file == 'cosby09':
        sample_data['wavenumbers'] = sample_data['wavenumbers'].add(36185)

    wns = sample_data['wavenumbers'].to_numpy()
    ins = sample_data['intensities'].to_numpy()
    ins /= max(ins)

    return wns, ins

def plot_line(data: list[tuple], colors: list[str], labels: list[str]) -> None:
    '''
    Plots wavenumber vs. intensity for line data.

    Args:
        data (list[tuple]): (wavenumbers, intensities)
        colors (list[str]): desired colors
        labels (list[str]): band labels
    '''

    for i, (wave, intn) in enumerate(data):
        if inp.VIB_BANDS[i][0] == 0:
            colr = 'b'
        else:
            colr = 'r'
        plt.stem(wave, intn, colr, markerfmt='', label=f'{labels[i]}')

def plot_sep_conv(data: list[tuple], colors: list[str], labels: list[str]) -> None:
    '''
    Plots wavenumber vs. intensity for convolved data.

    Args:
        data (list[tuple]): (wavenumbers, intensities)
        colors (list[str]): desired colors
        labels (list[str]): band labels
    '''

    for i, (wave, intn) in enumerate(data):
        if inp.VIB_BANDS[i][0] == 0:
            colr = 'b'
        else:
            colr = 'r'
        plt.plot(wave, intn, color=colr, label=f'{labels[i]}')

def plot_all_conv(data: tuple) -> None:
    '''
    Plots wavenumber vs. intensity for convolved data.

    Args:
        data (list[tuple]): (wavenumbers, intensities)
    '''

    plt.plot(data[0], data[1], label=f'Convolved Data: {inp.VIB_BANDS}')

def plot_samp(data: list[tuple], colors: list[str], labels: list[str]) -> None:
    '''
    Plots wavenumber vs. intensity for sample data.

    Args:
        data (list[tuple]): (wavenumbers, intensities)
        colors (list[str]): desired colors
        labels (list[str]): band labels
    '''

    for i, (wave, intn) in enumerate(data):
        plt.plot(wave, intn, colors[i], label=f'{labels[i]}')
