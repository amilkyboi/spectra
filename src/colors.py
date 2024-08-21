import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, to_hex

def get_colors(palette_size: str, bands: list[tuple[int, int]]) -> list[str]:
    """
    Returns a list of colors.
    """

    colors: list[str] = []

    match palette_size:
        case "small":
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        case "medium":
            palette: list[tuple] = plt.cycler("color", plt.cm.tab20c.colors).by_key()["color"]
            colors               = [to_hex(color) for color in palette]
        case "large":
            cmap:      Colormap  = plt.get_cmap("rainbow")
            num_bands: int       = len(bands)
            colors               = [to_hex(cmap(i / num_bands)) for i in range(num_bands)]
        case _:
            raise ValueError("ERROR: invalid palette size.")

    return colors
