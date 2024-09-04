# module colors
"""
Provides a list of colors based on the number of bands to be plotted.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, to_hex


def get_colors(bands: list[tuple[int, int]]) -> list[str]:
    """
    Returns a list of colors.
    """

    num_bands: int = len(bands)

    colors_small: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    palette: list[tuple] = plt.cycler("color", plt.cm.tab20c.colors).by_key()["color"]
    colors_medium: list[str] = [to_hex(color) for color in palette]

    if num_bands < len(colors_small):
        return colors_small

    if num_bands < len(colors_medium):
        return colors_medium

    cmap: Colormap = plt.get_cmap("rainbow")
    colors_large: list[str] = [to_hex(cmap(i / num_bands)) for i in range(num_bands)]

    return colors_large
