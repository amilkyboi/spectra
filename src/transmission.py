# module transmission
"""
Contains functions for modeling transmission.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # pylint: disable = unused-import

import constants as cn

plt.style.use(["science", "grid"])

def calc_density_cm(pressure_torr, temp):
    pressure_pa = pressure_torr * 101325 / 760

    return pressure_pa / (cn.BOLTZ * temp) / 1e6

vib_qn_up_max  = 3
vib_qn_lo_list = np.arange(0, 19)
length         = 1                                      # [cm]
temperature    = 3000                                   # [K]
pressure       = 1                                      # [Torr]
density        = calc_density_cm(pressure, temperature) # [cm^-3]
cross_section  = 14.2e-18                               # [cm^2]

# CS from: Absorption cross section of molecular oxygen, Lu, Chen
# N * L           ~ 10^15  [cm^-2]
# FC * CS         ~ 10^-18 [cm^2]
# FC * CS * N * L ~ 10^-3  [-]
# alpha = FC * CS * N
# I/I_0 = 1 - alpha * L

print(f"Total number density:            {density:10.4e} [cm^-3]")
print(f"Cross section at 295K, 143.26nm: {cross_section:10.4e} [cm^2]")

df      = pd.read_csv("../data/molecular_constants/o2.csv", index_col=0)
consts  = df.loc["x3sg"]
fc_data = np.loadtxt("../data/franck-condon/o2.csv", delimiter=',')

def vibrational_term(vib_qn_lo):
    return (consts["w_e"]   * (vib_qn_lo + 0.5)    -
            consts["we_xe"] * (vib_qn_lo + 0.5)**2 +
            consts["we_ye"] * (vib_qn_lo + 0.5)**3 +
            consts["we_ze"] * (vib_qn_lo + 0.5)**4)

total_partition = (np.exp(-vibrational_term(vib_qn_lo_list) *
                          cn.PLANC * cn.LIGHT / (cn.BOLTZ * temperature))).sum()

def number_density(vib_qn_lo):
    return (density * np.exp(-vibrational_term(vib_qn_lo) * cn.PLANC * cn.LIGHT /
                             (cn.BOLTZ * temperature)) / total_partition)

x = np.linspace(vib_qn_lo_list.min(), vib_qn_lo_list.max(), 1000)

def plot_text(func):
    for level in vib_qn_lo_list:
        plt.text(level + 0.1, func(level), f"{level}")

plt.plot(vib_qn_lo_list, number_density(vib_qn_lo_list), 'o')
plt.plot(x, number_density(x), label="Boltzmann Distribution")
plot_text(number_density)
plt.xlabel("Ground State Vibrational Level, $v\'\'$ [-]")
plt.ylabel("Vibrational Number Density, $N_v$ [-]")
plt.legend()
plt.show()

def get_fc(vib_qn_lo):
    return fc_data[vib_qn_up][vib_qn_lo]

for vib_qn_up in range(0, vib_qn_up_max):
    plt.plot(vib_qn_lo_list, get_fc(vib_qn_lo_list), "-o", label=f"$v\' = {vib_qn_up}$")

plot_text(get_fc)
plt.xlabel("Ground State Vibrational Level, $v\'\'$ [-]")
plt.ylabel("Franck-Condon Factor, FCF [-]")
plt.legend()
plt.show()

def factor(vib_qn):
    return get_fc(vib_qn) * cross_section * number_density(vib_qn) * length

for vib_qn_up in range(0, vib_qn_up_max):
    plt.plot(vib_qn_lo_list, factor(vib_qn_lo_list), "-o", label=f"$v\' = {vib_qn_up}$")

plot_text(factor)
plt.xlabel("Ground State Vibrational Level, $v\'\'$ [-]")
plt.ylabel("Factor, $\\sigma NL$ [-]")
plt.legend()
plt.show()

def transmission(vib_qn_lo):
    return np.exp(-factor(vib_qn_lo))

for vib_qn_up in range(0, vib_qn_up_max):
    plt.plot(vib_qn_lo_list, transmission(vib_qn_lo_list), "-o", label=f"$v\' = {vib_qn_up}$")

plot_text(transmission)
plt.xlabel("Ground State Vibrational Level, $v\'\'$ [-]")
plt.ylabel("Transmission, $I/I_0$ [-]")
plt.legend()
plt.show()
