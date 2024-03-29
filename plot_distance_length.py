#!/usr/bin/env python

"""
Script to create plot of chromosome distance to splindle equator & spindle length vs time relative to anaphase onset.
See main() for input requirements.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire
from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap, Normalize

from util_colormap import get_continuous_cmap

DEFAULT_HEX_LIST = [
    "#FEFAE0",
    "#FFE66D",
    "#FCBF49",
    "#FFBA08",
    "#FAA307",
    "#F48C06",
    "#E85D04",
    "#DC2F02",
    "#D00000",
    "#9D0208",
    "#6A040F",
    "#370617",
    "#03071E",
]
DEFAULT_PROPORTION_LIST = [4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1]


def clean(x: object) -> object:
    """Clean a cell value.
    In this case, if it's a string, we will ignore it (i.e. make it nan).

    Parameters
    ----------
    x : object
        Value of a dataframe cell

    Returns
    -------
    object
        nan if it's a string, it's original value otherwise.
    """
    if isinstance(x, str):
        return np.nan
    return x


def read_file(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """Read input distance or length file.
    This function will also preprocess the data (clean, set index, filter).

    Parameters
    ----------
    filepath : str
        The path to the input file.
    sheet_name : str [optional]
        The sheet name to read, by default the first sheet.

    Returns
    -------
    pd.DataFrame
        The data from the file.
    """
    if sheet_name is None:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.set_index(df.columns[0])
    df = df.applymap(clean)
    return df

def half(x: object) -> object:
    """Half the value of a cell if it's not nan.x

    Parameters
    ----------
    x : object
        Value of the cell, can be nan

    Returns
    -------
    object
        Half of the original value if not nan.x
    """
    if not np.isnan(x):
        return x / 2
    return x


def get_means_error(
    df: pd.DataFrame, error_type: str = "SD"
) -> Tuple[pd.Series, pd.Series]:
    """Compute the mean and standard deviation of the data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing distance/length data.
    error_type : str
        Can be either "SD" or "SEM" (standard dev. or standard means of err.)

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Tuple: means, standard deviation
    """
    means = df.mean(axis=1, skipna=True)
    if error_type is None or error_type.lower() == "none":
        err = None
    elif error_type == "SD":
        err = df.std(axis=1, skipna=True)
    elif error_type == "SEM":
        err = df.sem(axis=1, skipna=True)
    return means, err


def get_distance_cmap(
    hex_list: List[str], proportion_list: List[int]
) -> LinearSegmentedColormap:
    """Create a colormap for the distance scatter plot.py

    Parameters
    ----------
    hex_list : List[str]
        List of hex colors for the plot.
        The list should only contain the extreme color up to the middle color.
        This function will automatically mirror the colors to make a symmetrical colormap.
    proportion_list : List[int]
        List of proportions of the colors.
        This determines how much bigger is the area for each color relative to the other colors.
        The length of this list should be the same as the hex_list.

    Returns
    -------
    LinearSegmentedColormap
        Generated color map.
    """
    hex_list = hex_list + hex_list[:-1][::-1]
    if proportion_list is None:
        float_list = None
    else:
        proportion_list = proportion_list + proportion_list[:-1][::-1]
        float_list = [0.0]
        for i in proportion_list[:-1]:
            float_list.append(float_list[-1] + i / (sum(proportion_list)))
        float_list[-1] = 1.0
    distance_cmap = get_continuous_cmap(hex_list, float_list)
    return distance_cmap


def create_plot(
    df_distance: pd.DataFrame,
    distance_means: pd.Series,
    distance_stds: pd.Series,
    d_line_width: float,
    d_line_color: str,
    d_error_color: str,
    d_size: int,
    d_border: str,
    d_means_size: int,
    d_err_size: int,
    df_length: pd.DataFrame,
    half_length: bool,
    length_means: pd.Series,
    length_stds: pd.Series,
    l_line_color: str,
    l_error_color: str,
    l_means_size: int,
    l_err_size: int,
    distance_cmap: LinearSegmentedColormap,
    distance_min: int,
    distance_max: int,
    width: float,
    height: float,
    show_colorbar: bool,
    cbar_ticks_size: float,
    show_grid: bool,
    xlim: Tuple[int, int],
    ylim: Tuple[int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    title_size: float,
    xlabel_size: float,
    ylabel_size: float,
    xticks_size: float,
    yticks_size: float,
    xticks_interval: float,
    yticks_interval: float,
    marker,
) -> pyplot:
    """Create plot and save it to a file.

    Parameters
    ----------
    df_distance : pd.DataFrame
        DataFrame containing the distance data.
    distance_means : pd.Series
        The distance means.
    distance_stds : pd.Series
        The distance standard deviations.
    d_line_width : float,
        The width of distance lines.
    d_line_color : str
        Color for the distance means.
    d_error_color : str
        Color for the distance std. dev.
    d_size : int
        Size of the scatter points of the distance.
    d_border : str
        Border color for the scatter points of the distance.
    d_means_size : float
        Line size for distance means.
    d_err_size : float
        Line size for length error.
    df_length : pd.DataFrame
        DataFrame containing the length data.
    half_length : bool
        Whether to plot length as half & symmetrical.
    length_means : pd.Series
        The length means.
    length_stds : pd.Series
        The length standard deviations.
    l_line_color : str
        Color for the length means.
    l_error_color : str
        Color for the length std. dev.
    l_means_size : float
        Line size for length means.
    l_err_size : float
        Line size for length error.
    distance_cmap : LinearSegmentedColormap
        Colormap for the scatterplot.
    distance_min : int
        Minimum distance for the colormap.
    distance_max : int
        Maximum distance for the colormap.
    width : float
        The width of the image.
    height : float
        The height of the image.
    show_colorbar : bool
        Whether to display colorbar in the image.
    cbar_ticks_size: float
        Size of colorbar ticks
    show_grid : bool
        Whether to display grid in the image.
    x_lim : Tuple[int, int]
        [min, max] range of the x axis.
    y_lim : Tuple[int, int]
        [min, max] range of the y axis.
    title: str
        Title of the plot.
    xlabel: str
        Label of the x-axis.
    ylabel: str
        Label of the y-axis.
    title_size: float
        Size for title.
    xlabel_size: float
        Size for x label.
    ylabel_size: float
        Size for y label.
    xticks_size: float
        Size for x tick labels
    yticks_size: float
        Size for y tick labels
    xticks_interval: float
        Interval between x ticks
    yticks_interval: float
        Interval between y ticks
    marker: str
        Marker style

    Returns
    -------
    pyplot
        Generated plot.
    """

    fig, ax = plt.subplots(figsize=(width, height))

    # distance
    if df_distance is not None:
        norm = Normalize(vmin=distance_min, vmax=distance_max)
        y = df_distance.index
        for i in range(len(df_distance.columns)):
            x = df_distance[df_distance.columns[i]]
            if d_line_width > 0:
                for j in range(len(x)-1):
                    ax.plot(x.iloc[j:j+2], 
                            color=distance_cmap(norm(x.iloc[j])), 
                            linewidth=d_line_width)
            sc = ax.scatter(
                y,
                x,
                c=x,
                label=df_distance.columns[i],
                cmap=distance_cmap,
                marker=marker,
                s=d_size,
                vmin=distance_min,
                vmax=distance_max,
                edgecolors=d_border,
                zorder=4
            )
        # means & std
        if distance_means is not None:
            plt.plot(
                df_distance.index,
                distance_means,
                color=d_line_color,
                linewidth=d_means_size,
                zorder=5
            )
            if distance_stds is not None:
                plt.fill_between(
                    df_distance.index,
                    distance_means - distance_stds,
                    distance_means + distance_stds,
                    color=d_error_color,
                    alpha=0.2,
                    zorder=3
                )

    # length
    if df_length is not None:
        plt.plot(
            df_length.index,
            length_means,
            color=l_line_color,
            linewidth=l_means_size,
            zorder=2
        )
        if length_stds is not None:
            plt.fill_between(
                df_length.index,
                length_means - length_stds,
                length_means + length_stds,
                color=l_error_color,
                alpha=0.2,
                zorder=1
            )
        if half_length:
            plt.plot(
                df_length.index,
                -length_means,
                color=l_line_color,
                linewidth=l_means_size,
                zorder=2
            )
            if length_stds is not None:
                plt.fill_between(
                    df_length.index,
                    -length_means - length_stds,
                    -length_means + length_stds,
                    color=l_error_color,
                    alpha=0.2,
                    zorder=1
                )

    plt.title(title, size=title_size)
    plt.ylabel(ylabel, size=ylabel_size)
    plt.xlabel(xlabel, size=xlabel_size)

    plt.ylim(ylim)
    plt.xticks(np.arange(xlim[0], xlim[1]+xticks_interval, xticks_interval), 
                fontsize=xticks_size)
    plt.yticks(np.arange(ylim[0], ylim[1]+yticks_interval, yticks_interval), 
                fontsize=yticks_size)

    if df_distance is not None and show_colorbar:
        cbar = plt.colorbar(sc)
        cbar.ax.tick_params(labelsize=cbar_ticks_size) 
    if show_grid:
        plt.grid()

    return plt


def main(
    file_distance: str,
    file_length: str = None,
    half_length: bool = True,
    outfile: str = "foo.jpg",
    min_time: int = -500,
    max_time: int = 0,
    distance_color: str = "orange",
    distance_error_color: str = "orange",
    length_color: str = "green",
    length_error_color: str = "green",
    distance_size: int = 40,
    distance_border: str = "face",
    plot_width: int = 20,
    plot_height: int = 10,
    show_colorbar: bool = False,
    show_grid: bool = False,
    hex_list: str = None,
    proportion_list: List[int] = None,
    distance_min: int = -3,
    distance_max: int = 3,
    ylim_min: int = -5,
    ylim_max: int = 5,
) -> None:
    """Create a plot of chromosome distance to splindle equator & spindle length vs time relative to anaphase onset.

    Parameters
    ----------
    file_distance : str
        Path to excel file containing the data for chromosome distance to splindle equator vs time relative to anaphase onset.
        Rows = time relative to anaphase onset; columns=data points; cell values = chromosome distance to splindle equator.
        All values that are not proper numbers (e.g. ending with '*') will be ignored.
        Make sure to write the path in between quotes.
    file_length : str, optional
        Path to excel file containing the data for spindle length vs time relative to anaphase onset, by default None.
        Rows = time relative to anaphase onset; columns=data points; cell values = spindle length.
        All values that are not proper numbers (e.g. ending with '*') will be ignored.
        Make sure to write the path in between quotes.
    half_length : bool, optional
        Whether to plot the length by half and symmetrical, by default True.
    outfile : str, optional
        Path to the output image, by default "foo.jpg".
        Make sure to write the path in between quotes. The file should end with either ".jpg" or ".png".
    min_time : int, optional
        The minimum time relative to anaphase onset that should be included, by default -500.
    max_time : int, optional
        The maximum time relative to anaphase onset that should be included, by default 0.
    distance_color : str, optional
        The color for distance means, by default 'orange'. It can accept standard colors.
    distance_error_color : str, optional
        The color for distance std dev, by default 'orange'. It can accept standard colors.
    length_color : str, optional
        The color for distance means, by default 'green'. It can accept standard colors.
    length_error_color : str, optional
        The color for distance std dev, by default 'green'. It can accept standard colors.
    distance_size : int, optional
        The size for distance data points, by default 40.
    distance_border : str, optional
        The border color for distance data points, by default "face". It can accept standard colors, "face" means the same color as the point.
    plot_width : int, optional
        The width of the resulting plot image, by default 20.
    plot_height : int, optional
        The width of the resulting plot image, by default 10.
    show_colorbar : bool, optional
        Use this option to show colorbar in the plot, by default False.
    show_grid : bool, optional
        Use this option to show grid in the plot, by default False.
    hex_list : str, optional
        List of colors (in hexadecimal code) to modify the colormap of the scatter plot, by default None.
        The resulting colormap will be a symmetrical colormap.
        For example, if -hex_list="#FEFAE0,#03071E", then the color map will be: #FEFAE0,#03071E,#FEFAE0
    proportion_list : List[int], optional
        List of proportion of how big each point (except for the last one) on the hex list should be, by default None.
        For example, if -hex_list="#FEFAE0,#03071E" -proportion_list="1,2",
        then the resulting colormap will be #FEFAE0,#03071E,#FEFAE0, with the area for #03071E being twice as big as the areas of the #FEFAE0s.
    distance_min : int, optional
        The lowest distance for the colormap, by default -3.
        This determines what color of each values;
        the extremes being the distance_min and distance_max, and the middle color being the middle between these two values.
    distance_max : int, optional
        The maximum distance for the colormap, by default 3.
        This determines what color of each values;
        the extremes being the distance_min and distance_max, and the middle color being the middle between these two values.
    ylim_min : int, optional
        The minimum value of the y axis, by default -5.
    ylim_max : int, optional
        The maximum value of the y axis, by default 5.
    """

    df_distance = read_file(file_distance)
    df_distance = df_distance.loc[
        (df_distance.index >= min_time) & (df_distance.index <= max_time)
    ]
    distance_means, distance_stds = get_means_error(df_distance)

    if file_length is not None:
        df_length = read_file(file_length)
        df_length = df_length.loc[
            (df_length.index >= min_time) & (df_length.index <= max_time)
        ]
        if half_length:
            df_length = df_length.applymap(half)
        length_means, length_stds = get_means_error(df_length)
    else:
        df_length = None
        length_means = None
        length_stds = None

    if hex_list is None:
        hex_list = DEFAULT_HEX_LIST
        proportion_list = DEFAULT_PROPORTION_LIST
    else:
        hex_list = hex_list.split(",")
    distance_cmap = get_distance_cmap(hex_list, proportion_list)

    title = "JDU233 in utero congression"
    ylabel = "Distance to spindle equator (\u03BCm)"
    xlabel = "Time relative to Anaphase Onset (s)"

    plt = create_plot(
        df_distance,
        distance_means,
        distance_stds,
        distance_color,
        distance_error_color,
        distance_size,
        distance_border,
        2,
        2,
        df_length,
        half_length,
        length_means,
        length_stds,
        length_color,
        length_error_color,
        2,
        2,
        distance_cmap,
        distance_min,
        distance_max,
        plot_width,
        plot_height,
        show_colorbar,
        10,
        show_grid,
        (min_time, max_time),
        (ylim_min, ylim_max),
        title,
        xlabel,
        ylabel,
        10,
        10,
        10,
        10,
        10,
        100.0,
        1.0,
        "o",
    )

    plt.savefig(outfile)
    print(f"Plot saved to {outfile}")


if __name__ == "__main__":
    Fire(main)
