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
from matplotlib import collections, pyplot
from matplotlib.colors import LinearSegmentedColormap, Normalize

from util_colormap import get_continuous_cmap

from plot_distance_length import *

def create_plot(
    df_distance: pd.DataFrame,
    df_threshold: pd.DataFrame,
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
    df_threshold : pd.DataFrame
        DataFrame containing the threshold percentage for coloring.
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
        y = df_distance.index
        prev_x = None
        prev_threshold = None
        prev_time_stamp = None
        for time_stamp in y:
            threshold = df_threshold.at[time_stamp, df_threshold.columns[0]] * 100
            if not np.isnan(threshold):
                x = df_distance.loc[time_stamp].to_numpy()
                sc = ax.scatter(
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
                if d_line_width > 0 and prev_x is not None:
                    data_dict = {}
                    for i, (x_0, x_1) in enumerate(zip(prev_x, x)):
                        data_dict[i] = [x_0, x_1]
                    df_line = pd.DataFrame(data_dict, index=[prev_time_stamp, time_stamp])
                    ax.plot(df_line,
                            color=distance_cmap(prev_threshold/100), 
                            linewidth=d_line_width)
                prev_x = x
                prev_threshold = threshold
                prev_time_stamp = time_stamp
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


