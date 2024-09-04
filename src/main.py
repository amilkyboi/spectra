# module main
"""
Testing GUI functionality.
"""

import warnings
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from typing import Callable

import numpy as np
import pandas as pd
from pandastable import Table
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

import plot
from simtype import SimType
from colors import get_colors
from molecule import Molecule
from simulation import Simulation

# I think an internal function within pandastable is using .fillna or a related function that emits
# "FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will
# change in a future version."
#
# The fixes in the linked thread don't seem to work, which is why I think the issue is internal to
# pandastable itself. For now, just disable the warning.
# https://stackoverflow.com/questions/77900971/pandas-futurewarning-downcasting-object-dtype-arrays-on-fillna-ffill-bfill
pd.set_option("future.no_silent_downcasting", True)
# More warnings for d_type conversions that aren't yet fixed in a release build of pandastable, see:
# https://github.com/dmnfarrell/pandastable/issues/251
warnings.simplefilter(action="ignore", category=FutureWarning)

DEFAULT_TEMPERATURE: float = 300.0  # [K]
DEFAULT_PRESSURE: float = 101325.0  # [Pa]

DEFAULT_BANDS: str = "0-0"
DEFAULT_PLOTTYPE: str = "Line"
DEFAULT_SIMTYPE: str = "Absorption"


def set_axis_labels(ax: Axes) -> None:
    secax = ax.secondary_xaxis("top", functions=(plot.wavenum_to_wavelen, plot.wavenum_to_wavelen))
    secax.set_xlabel("Wavenumber, $\\nu$ [cm$^{-1}$]")

    ax.set_xlabel("Wavelength, $\\lambda$ [nm]")
    ax.set_ylabel("Intensity, Arbitrary Units [-]")


def create_figure() -> tuple[Figure, Axes]:
    fig: Figure = Figure()
    axs: Axes = fig.add_subplot(111)

    # Set the left x-limit to something greater than zero so the secondary axis doesn't encounter a
    # divide by zero error before any data is actually plotted
    axs.set_xlim(100, 200)

    set_axis_labels(axs)

    return fig, axs


class MolecularSimulationGUI:
    """
    The GUI.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root

        self.root.title("Diatomic Molecular Simulation")

        # Center the window on the screen
        screen_width: int = self.root.winfo_screenwidth()
        screen_height: int = self.root.winfo_screenheight()

        window_height: int = 600
        window_width: int = 1200

        x_offset: int = int((screen_width / 2) - (window_width / 2))
        y_offset: int = int((screen_height / 2) - (window_height / 2))

        self.root.geometry(f"{window_width}x{window_height}+{x_offset}+{y_offset}")

        self.create_widgets()

    def create_widgets(self) -> None:
        """
        Create widgets.
        """

        # Frames -----------------------------------------------------------------------------------

        # Frame for input boxes
        self.frame_input: ttk.Frame = ttk.Frame(self.root)
        self.frame_input.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.frame_input_entries: ttk.Frame = ttk.Frame(self.frame_input)
        self.frame_input_entries.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        self.frame_input_combos: ttk.Frame = ttk.Frame(self.frame_input)
        self.frame_input_combos.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5)

        # Frame for the table and plot
        self.frame_main: ttk.Frame = ttk.Frame(self.root)
        self.frame_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame for the table
        self.frame_table: ttk.Frame = ttk.Frame(self.frame_main)
        self.frame_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for the plot
        self.frame_plot: ttk.Frame = ttk.Frame(self.frame_main)
        self.frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Entries ----------------------------------------------------------------------------------

        self.temperature = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        ttk.Label(self.frame_input_entries, text="Temperature [K]:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Entry(self.frame_input_entries, textvariable=self.temperature).grid(
            row=0, column=1, padx=5, pady=5
        )

        self.pressure = tk.DoubleVar(value=DEFAULT_PRESSURE)
        ttk.Label(self.frame_input_entries, text="Pressure [Pa]:").grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )
        ttk.Entry(self.frame_input_entries, textvariable=self.pressure).grid(
            row=0, column=3, padx=5, pady=5
        )

        self.band_ranges = tk.StringVar(value=DEFAULT_BANDS)
        ttk.Label(self.frame_input_entries, text="Band Ranges (format: v'-v''):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Entry(self.frame_input_entries, textvariable=self.band_ranges, width=50).grid(
            row=1, column=1, columnspan=3, padx=5, pady=5
        )

        # Comboboxes -------------------------------------------------------------------------------

        self.simulation = tk.StringVar(value=DEFAULT_SIMTYPE)
        ttk.Label(self.frame_input_combos, text="Simulation Type:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Combobox(
            self.frame_input_combos, textvariable=self.simulation, values=("Absorption", "Emission")
        ).grid(row=0, column=1, padx=5, pady=5)

        self.plot_type = tk.StringVar(value=DEFAULT_PLOTTYPE)
        ttk.Label(self.frame_input_combos, text="Plot Type:").grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )
        ttk.Combobox(
            self.frame_input_combos,
            textvariable=self.plot_type,
            values=("Line", "Line Info", "Convolution", "Band Info"),
        ).grid(row=0, column=3, padx=5, pady=5)

        # Button -----------------------------------------------------------------------------------

        ttk.Button(self.frame_input, text="Run Simulation", command=self.run_simulation).pack()

        # Notebook ---------------------------------------------------------------------------------

        # Holds multiple tables, one for each specified vibrational band
        self.notebook = ttk.Notebook(self.frame_table)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Table ------------------------------------------------------------------------------------

        # Initialize the table with an empty dataframe so that nothing is shown until a simulation
        # is run by the user
        tab_frame = ttk.Frame(self.notebook)
        table = Table(
            tab_frame,
            dataframe=pd.DataFrame(),
            showtoolbar=True,
            showstatusbar=True,
            editable=False,
        )
        table.show()
        self.notebook.add(tab_frame, text="Band v'-v''")

        # Plot -------------------------------------------------------------------------------------

        # Draw the initial figure and axes with no data present
        self.fig: Figure
        self.axs: Axes
        self.fig, self.axs = create_figure()
        self.plot_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Show the matplotlib toolbar
        self.toolbar: NavigationToolbar2Tk = NavigationToolbar2Tk(self.plot_canvas)

        # Map plot types to functions
        self.map_functions: dict[str, Callable] = {
            "Line": plot.plot_line,
            "Line Info": plot.plot_line_info,
            "Convolution": plot.plot_conv,
            "Band Info": plot.plot_band_info,
        }

    def parse_band_ranges(self) -> list[tuple[int, int]]:
        """
        Convert comma-separated user input of the form 0-0, 0-2, etc. into valid vibrational bands.
        """

        band_ranges_str: str = self.band_ranges.get()

        bands: list[tuple[int, int]] = []

        for range_str in band_ranges_str.split(","):
            range_str: str = range_str.strip()

            if "-" in range_str:
                try:
                    lower_band: int
                    upper_band: int
                    lower_band, upper_band = map(int, range_str.split("-"))
                    bands.append((lower_band, upper_band))
                except ValueError:
                    messagebox.showinfo("Info", f"Invalid band range format: {range_str}")
            else:
                messagebox.showinfo("Info", f"Invalid band range format: {range_str}")

        return bands

    def run_simulation(self) -> None:
        """
        Runs a simulation instance.
        """

        # Grab the temperature, pressure, and simulation type directly from the input fields
        temperature: float = self.temperature.get()
        pressure: float = self.pressure.get()
        # Convert to uppercase to use as a key for the SimType enum
        sim_type: str = self.simulation.get().upper()
        # Upper and lower vibrational bands
        bands: list[tuple[int, int]] = self.parse_band_ranges()

        molecule: Molecule = Molecule("o2", "o", "o")
        simulation: Simulation = Simulation(
            molecule,
            temperature,
            pressure,
            np.arange(0, 36),
            "b3su",
            "x3sg",
            bands,
            SimType[sim_type],
        )

        colors: list[str] = get_colors(bands)

        # Clear the previously plotted data and reset the axis labels
        self.axs.clear()
        set_axis_labels(self.axs)

        # Choose the plotting function based on the selected plot type
        plot_type: str = self.plot_type.get()
        plot_function: Callable | None = self.map_functions.get(plot_type)

        if plot_function:
            plot_function(self.axs, simulation, colors)
        else:
            messagebox.showinfo("Info", f"Plot type '{plot_type}' is not recognized.")

        self.axs.legend()
        self.plot_canvas.draw()

        # Clear previous tabs
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)

        # Get maximum intensity for the entire vibrational band range for normalization
        max_intensity: float = simulation.all_line_data()[1].max()

        # Each vibrational band has a separate tab associated with it, each tab gets updated
        # separately
        for i, band in enumerate(bands):
            data: list[dict[str, float | int | str]] = [
                {
                    "Wavenumber": 0.0,
                    "Intensity": 0.0,
                    "J'": line.rot_qn_up,
                    "J''": line.rot_qn_lo,
                    "Branch": line.branch_name,
                    "N'": line.branch_idx_up,
                    "N''": line.branch_idx_lo,
                }
                for line in simulation.vib_bands[i].lines
            ]

            df = pd.DataFrame(data)
            df["Wavenumber"] = simulation.vib_bands[i].wavenumbers_line()
            df["Intensity"] = simulation.vib_bands[i].intensities_line() / max_intensity

            tab_frame: ttk.Frame = ttk.Frame(self.notebook)
            table: Table = Table(
                tab_frame, dataframe=df, showtoolbar=True, showstatusbar=True, editable=False
            )
            table.show()
            self.notebook.add(tab_frame, text=f"Band {band[0]}-{band[1]}")


def main() -> None:
    """
    Runs the program.
    """

    root: tk.Tk = tk.Tk()
    MolecularSimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
