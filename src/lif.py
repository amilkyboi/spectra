# module test
'''
Testing a better implementation of LIF simulation.
'''

import itertools

import scienceplots # pylint: disable=unused-import
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import convolution as conv
import initialize as init
import constants as cn
import input as inp
import energy

plt.style.use(['science', 'grid'])
plt.rcParams.update({'font.size': inp.FONT_SIZE[1]})

def selection_rules(gnd_rot_qn, ext_rot_qn):

    lines = []

    d_rot_qn = ext_rot_qn - gnd_rot_qn

    # For molecular oxygen, all transitions with even values of J'' are forbidden
    if gnd_rot_qn % 2 == 1:

        # Selection rules for the R branch
        if d_rot_qn == 1:
            for grnd_branch_idx, exct_branch_idx in itertools.product(range(1, 4), repeat=2):
                if grnd_branch_idx == exct_branch_idx:
                    lines.append(init.SpectralLine(gnd_rot_qn, ext_rot_qn, 'r',
                                                   grnd_branch_idx, exct_branch_idx, 0.0))
                if grnd_branch_idx > exct_branch_idx:
                    lines.append(init.SpectralLine(gnd_rot_qn, ext_rot_qn, 'rq',
                                                   grnd_branch_idx, exct_branch_idx, 0.0))

        # Selection rules for the P branch
        elif d_rot_qn == -1:
            for grnd_branch_idx, exct_branch_idx in itertools.product(range(1, 4), repeat=2):
                if grnd_branch_idx == exct_branch_idx:
                    lines.append(init.SpectralLine(gnd_rot_qn, ext_rot_qn, 'p',
                                                   grnd_branch_idx, exct_branch_idx, 0.0))
                if grnd_branch_idx < exct_branch_idx:
                    lines.append(init.SpectralLine(gnd_rot_qn, ext_rot_qn, 'pq',
                                                   grnd_branch_idx, exct_branch_idx, 0.0))

    for line in lines:
        if line.gnd_branch_idx == 1:
            line.predissociation = inp.PD_DATA['f1'][inp.PD_DATA['rot_qn'] == line.ext_rot_qn].iloc[0]
        elif line.gnd_branch_idx == 2:
            line.predissociation = inp.PD_DATA['f2'][inp.PD_DATA['rot_qn'] == line.ext_rot_qn].iloc[0]
        else:
            line.predissociation = inp.PD_DATA['f3'][inp.PD_DATA['rot_qn'] == line.ext_rot_qn].iloc[0]

    return np.array(lines)

class LinePlot:
    '''
    Each LinePlot is a separate vibrational band.
    '''

    def __init__(self, temp: float, pres: float, gnd_rot_qn: int, ext_rot_qn: int,
                 states: tuple[int, int]) -> None:
        self.temp         = temp
        self.pres         = pres
        self.gnd_rot_qn   = gnd_rot_qn
        self.ext_rot_qn   = ext_rot_qn
        self.states       = states

    def get_fc(self) -> float:
        '''
        From the global Franck-Condon data array, grabs the correct FC factor for the current
        vibrational transition.

        Args:
            fc_data (np.ndarray): global Franck-Condon array

        Returns:
            float: Franck-Condon factor for the current vibrational transition
        '''

        return inp.FC_DATA[self.states[0]][self.states[1]]

    def return_line(self):
        return selection_rules(self.gnd_rot_qn, self.ext_rot_qn)

    def get_line(self, max_fc: float) -> tuple[float, float]:
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

        # Find the valid spectral line
        lines = selection_rules(self.gnd_rot_qn, self.ext_rot_qn)

        # Get the wavenumber and intensity
        wns = np.array([line.wavenumber(band_origin, grnd_state, exct_state)
                        for line in lines])
        ins = np.array([line.intensity(band_origin, grnd_state, exct_state, self.temp)
                        for line in lines])

        # Find the ratio between the largest Franck-Condon factor and the current plot
        norm_fc = self.get_fc() / max_fc

        # This is normalization of the plot with respect to others
        ins *= norm_fc

        return wns, ins

def main():
    '''
    Runs the program.
    '''

    gnd_rot_qn = 21
    ext_rot_qn = 22

    ext_vib_qn = 7
    max_vib_qn = 12

    band_list = []
    for gnd_vib_qn in range(max_vib_qn + 1):
        band_list.append(LinePlot(inp.TEMP, inp.PRES, gnd_rot_qn, ext_rot_qn, (ext_vib_qn, gnd_vib_qn)))

    max_fc = max((band.get_fc() for band in band_list))

    line_data = [band.get_line(max_fc) for band in band_list]

    lines = np.array([band.return_line() for band in band_list]).flatten()

    wvnums = np.array([line[0] for line in line_data]).flatten()
    intens = np.array([line[1] for line in line_data]).flatten()
    intens /= intens.max()

    conv_wns, conv_ins = conv.convolved_data(wvnums, intens, inp.TEMP, inp.PRES, lines)

    # Grab a rainbow colormap from the built-in matplotlib cmaps
    cmap = plt.get_cmap('rainbow')
    num_lines = len(wvnums)

    # Assign each line a color, each being equally spaced within the colormap
    colors = [mcolors.to_hex(cmap(i / (num_lines - 1))) for i in range(num_lines)]

    _, axs = plt.subplots(1, 1, figsize=(inp.SCREEN_RES[0]/inp.DPI, inp.SCREEN_RES[1]/inp.DPI),
                          dpi=inp.DPI, sharex=True)

    for i, (wave, intn) in enumerate(zip(wvnums, intens)):
        _, stemlines, _ = axs.stem((1 / wave) * 1e7, intn, colors[i], markerfmt='')
        plt.setp(stemlines, 'linewidth', 3)

    axs.set_title(f"Initial Laser Excitation: $(v', v'') = ({ext_vib_qn}, {0})$, \
                    Emission: $v''_\\mathrm{{max}} = {max_vib_qn}$, $v''_\\mathrm{{min}} = 0$, \
                    Selected Line: $(N', N'') = ({ext_rot_qn}, {gnd_rot_qn})$")
    axs.set_ylabel('Normalized Intensity')

    # Convert from wavenumber to wavelength
    def wn2wl(wns):
        return (1 / wns) * 1e7

    # Add a secondary axis for wavelength
    secax = axs.secondary_xaxis('top', functions=(wn2wl, wn2wl))
    secax.set_xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')

    # axs[1].plot((1 / conv_wns) * 1e7, conv_ins)
    axs.set_xlabel('Wavelength $\\nu$, [nm]')

    plt.savefig(inp.PLOT_PATH, dpi=inp.DPI * 2)
    # plt.show()

if __name__ == '__main__':
    main()
