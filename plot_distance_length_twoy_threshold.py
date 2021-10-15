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
from plot_distance_length import (
    read_file,
    get_means_error,
    DEFAULT_HEX_LIST,
    DEFAULT_PROPORTION_LIST,
    half,
    get_distance_cmap,
)


def create_plot(
    df_distance: pd.DataFrame,
    df_threshold,
    distance_means: pd.Series,
    distance_stds: pd.Series,
    d_line_color: str,
    d_error_color: str,
    d_width: float,
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
    d_ylim: Tuple[int, int],
    l_ylim: Tuple[int, int],
    title: str,
    xlabel: str,
    d_ylabel: str,
    l_ylabel: str,
    title_size: float,
    xlabel_size: float,
    d_ylabel_size: float,
    l_ylabel_size: float,
    xticks_size: float,
    d_yticks_size: float,
    l_yticks_size: float,
    xticks_interval: float,
    d_yticks_interval: float,
    l_yticks_interval: float,
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
    d_line_color : str
        Color for the distance means.
    d_error_color : str
        Color for the distance std. dev.
    d_width : float
        Width of the distance line.
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

    if df_distance is not None and df_length is not None:
        axes = [ax, ax.twinx()]
        # Make some space on the right side for the extra y-axis.
        # fig.subplots_adjust(right=0.8)
        # Move the last y-axis spine over to the right by 20% of the width of the axes
        # axes[-1].spines['right'].set_position(('axes', 1.2))
        # To make the border of the right-most axis visible, we need to turn the frame
        # on. This hides the other plots, however, so we need to turn its fill off.
        axes[-1].set_frame_on(True)
        axes[-1].patch.set_visible(False)
    else:
        axes = [ax]

    # length
    if df_length is not None:
        cur_ax = axes[0]
        # if half_length:
        cur_ax.plot(
            df_length.index,
            length_means,
            color=l_line_color,
            linewidth=l_means_size,
            zorder=2,
        )
        if length_stds is not None:
            cur_ax.fill_between(
                df_length.index,
                length_means - length_stds,
                length_means + length_stds,
                color="none",
                alpha=0.2,
                facecolor=l_error_color,
                zorder=1,
            )
        if half_length:
            cur_ax.plot(
                df_length.index,
                -length_means,
                color=l_line_color,
                linewidth=l_means_size,
                zorder=2,
            )
            if length_stds is not None:
                cur_ax.fill_between(
                    df_length.index,
                    -length_means - length_stds,
                    -length_means + length_stds,
                    color="none",
                    alpha=0.2,
                    facecolor=l_error_color,
                    zorder=1,
                )
        cur_ax.set_ylabel(
            l_ylabel,
            size=l_ylabel_size,
        )
        cur_ax.set_ylim(l_ylim)
        cur_ax.set_yticks(
            np.arange(l_ylim[0], l_ylim[1] + l_yticks_interval, l_yticks_interval)
        )
        cur_ax.tick_params(axis="y", labelsize=l_yticks_size)

    # distance
    if df_distance is not None:
        cur_ax = axes[-1]
        # means & std
        if distance_means is not None:
            cur_ax.plot(
                df_distance.index,
                distance_means,
                linewidth=d_means_size,
                color=d_line_color,
                zorder=5,
            )
            if distance_stds is not None:
                cur_ax.fill_between(
                    df_distance.index,
                    distance_means - distance_stds,
                    distance_means + distance_stds,
                    color="none",
                    alpha=0.2,
                    facecolor=d_error_color,
                    zorder=3,
                )
        # scatter
        y = df_distance.index
        prev_x = None
        prev_threshold = None
        prev_time_stamp = None
        for time_stamp in y:
            threshold = df_threshold.at[time_stamp, df_threshold.columns[0]] * 100
            if not np.isnan(threshold):
                x = df_distance.loc[time_stamp].to_numpy()
                sc = cur_ax.scatter(
                    [time_stamp] * len(x),
                    x,
                    c=[threshold] * len(x),
                    label=df_distance.columns,
                    cmap=distance_cmap,
                    marker=marker,
                    s=d_size,
                    vmin=distance_min,
                    vmax=distance_max,
                    edgecolors=d_border,
                    zorder=4,
                )
                if d_width > 0 and prev_x is not None:
                    data_dict = {}
                    for i, (x_0, x_1) in enumerate(zip(prev_x, x)):
                        data_dict[i] = [x_0, x_1]
                    df_line = pd.DataFrame(data_dict, index=[prev_time_stamp, time_stamp])
                    ax.plot(df_line,
                            color=distance_cmap(prev_threshold/100), 
                            linewidth=d_width)
                prev_x = x
                prev_threshold = threshold
                prev_time_stamp = time_stamp
        cur_ax.set_ylabel(
            d_ylabel,
            size=d_ylabel_size,
        )
        cur_ax.set_ylim(d_ylim)
        cur_ax.set_yticks(
            np.arange(d_ylim[0], d_ylim[1] + d_yticks_interval, d_yticks_interval)
        )
        cur_ax.tick_params(axis="y", labelsize=d_yticks_size)

    plt.title(title, size=title_size)

    ax.set_xlabel(xlabel, size=xlabel_size)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + xticks_interval, xticks_interval))
    ax.tick_params(axis="x", labelsize=xticks_size)

    if df_distance is not None and show_colorbar:
        cbar = plt.colorbar(sc, orientation="horizontal")
        cbar.ax.tick_params(labelsize=cbar_ticks_size)
    if show_grid:
        plt.grid()

    return plt