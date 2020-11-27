""" Colormap creator & viewer.
Creator from https://github.com/KerryHalupka/custom_colormap."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values"""
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values"""
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map"""
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def get_colormap_plot(
    colormap: LinearSegmentedColormap, vmin: int = -3, vmax: int = 3
) -> Figure:
    """Create a plot image of the colormap

    Parameters
    ----------
    colormap : LinearSegmentedColormap
        The colormap
    vmin : int, optional
        Minimum value that the colormap represents, by default -3
    vmax : int, optional
        Maximum value that the colormap represents, by default 3

    Returns
    -------
    Figure
        The colormap plot.
    """
    a = np.outer(np.arange(vmin, vmax, 0.01), np.ones(1)).transpose()
    fig, ax = plt.subplots(1, 1, figsize=(4, 0.5))
    ax.imshow(
        a,
        aspect="auto",
        cmap=colormap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[vmin, vmax, -1, 1],
    )
    x_label_list = [vmin, (vmax+vmin)/2, vmax]
    ax.set_xticks([vmin, (vmax+vmin)/2, vmax])
    ax.set_xticklabels(x_label_list)
    ax.set_yticks([])
    return fig
