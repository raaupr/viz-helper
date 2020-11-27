"""GUI for plot_distance_length.py"""

import os

import streamlit as st

from plot_distance_length import (
    DEFAULT_HEX_LIST,
    DEFAULT_PROPORTION_LIST,
    create_plot,
    get_distance_cmap,
    get_means_error,
    half,
    read_file,
)
from util_colormap import get_colormap_plot

st.title("Visualize Chromosome Distance & Splindle Length vs Time")

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("Customization options")

# -- Get data
st.sidebar.header("Distance")
df_distance = None
is_plot_distance = st.sidebar.checkbox("Plot distance")
is_plot_distance_means = True
distance_error_type = None
if is_plot_distance:
    fin_distance = st.sidebar.file_uploader(
        "Select the chromosome-spindle distance file (.xlsx file only!)", type="xlsx"
    )
    if fin_distance:
        df_distance = read_file(fin_distance)
        st.info(
            f"Reading distance file successful: found {len(df_distance.columns)} data points."
        )
        is_plot_distance_means = st.sidebar.checkbox(
            "Plot distance means", value="True"
        )
        distance_error_type = st.sidebar.selectbox(
            "Error bar type", ["None", "SD", "SEM"], key="distance_error_type"
        )
        if distance_error_type == "None":
            distance_error_type = None
    else:
        st.warning("\u2190 Please select the distance dataset file.")
        st.sidebar.warning("Please select the distance dataset file.")
st.sidebar.header("Length")
df_length = None
is_plot_length = st.sidebar.checkbox("Plot length")
is_half_length = False
length_error_type = None
if is_plot_length:
    fin_length = st.sidebar.file_uploader(
        "Select the spindle length file (.xlsx file only!)", type="xlsx"
    )
    if fin_length:
        df_length = read_file(fin_length)
        st.info(
            f"Reading length file successful: found {len(df_length.columns)} data points."
        )
        is_half_length = st.sidebar.checkbox(
            "Make length plot symmetrical w.r.t. the center", value=True
        )
        length_error_type = st.sidebar.selectbox(
            "Error bar type", ["None", "SD", "SEM"], key="length_error_type"
        )
        if length_error_type == "None":
            length_error_type = None
    else:
        st.warning("\u2190 Please select the length dataset file.")
        st.sidebar.warning("Please select the length dataset file.")
if not is_plot_distance and not is_plot_length:
    st.write("\u2190 Please select what to plot.")


if (is_plot_distance and df_distance is not None) or (
    is_plot_length and df_length is not None
):
    # ---- CUSTOMIZATION
    st.sidebar.header("Ranges:")
    st.sidebar.subheader("x-axis")
    if df_distance is not None:
        min_default = df_distance.index[0]
        max_default = df_distance.index[-1]
        if df_length is not None:
            min_default = min(min_default, df_length.index[0])
            max_default = max(max_default, df_length.index[-1])
    else:
        min_default = df_length.index[0]
        max_default = df_length.index[-1]
    min_time = st.sidebar.number_input(
        "Min time:", min_value=min_default, max_value=max_default
    )
    max_time = st.sidebar.number_input(
        "Max time:", min_value=min_default, max_value=max_default, value=max_default
    )
    st.sidebar.subheader("y-axis")
    ylim_min = st.sidebar.number_input("Min value:", value=-5)
    ylim_max = st.sidebar.number_input("Max value:", value=5)

    st.sidebar.header("Texts:")
    title = st.sidebar.text_input("Title:", value="JDU233 in utero congression")
    title_size = st.sidebar.number_input("Title size", value=15)
    xlabel = st.sidebar.text_input(
        "x-axis label:", value="Time relative to Anaphase Onset (s)"
    )
    xlabel_size = st.sidebar.number_input("x-axis label size", value=10)
    xticks_size = st.sidebar.number_input("x-axis tick labels size", value=10)
    ylabel = st.sidebar.text_input(
        "y-axis label:", value="Distance to spindle equator (\u03BCm)"
    )
    ylabel_size = st.sidebar.number_input("y-axis label size", value=10)
    yticks_size = st.sidebar.number_input("y-axis tick labels size", value=10)

    st.sidebar.header("Sizes:")
    plot_width = st.sidebar.number_input("Plot width:", value=20)
    plot_height = st.sidebar.number_input("Plot height:", value=10)
    distance_size = 40
    d_means_size, d_error_size, l_means_size, l_error_size = 2, 2, 2, 2
    marker = 'o'
    if df_distance is not None:
        marker = st.sidebar.text_input("Marker type for distance points", value="o")
        st.sidebar.markdown(
            "See [here](https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/marker_reference.html) for possible marker types"
        )
        distance_size = st.sidebar.number_input("Size of distance points", value=40)
    if is_plot_distance_means:
        d_means_size = st.sidebar.number_input("Distance means line width", value=2)
        if distance_error_type is not None:
            d_error_size = st.sidebar.number_input("Distance error line width", value=2)
    if is_plot_length:
        l_means_size = st.sidebar.number_input("Length means line width", value=2)
        if length_error_type is not None:
            l_error_size = st.sidebar.number_input("Length error line width", value=2)

    st.sidebar.header("Elements:")
    if df_distance is not None:
        show_colorbar = st.sidebar.checkbox("Show color bar")
    else:
        show_colorbar = False
    show_grid = st.sidebar.checkbox("Show grid")

    st.sidebar.header("Colors:")
    if df_distance is not None and is_plot_distance_means:
        distance_color = st.sidebar.color_picker(
            "Distance means & std:", value="#FFA500"
        )
    else:
        distance_color = None
    if df_length is not None:
        length_color = st.sidebar.color_picker("Length means & std:", value="#008000")
    else:
        length_color = None
    distance_border = "face"
    if df_distance is not None:
        use_distance_border = st.sidebar.checkbox(
            "Use border on distance marker points"
        )
        if use_distance_border:
            distance_border = st.sidebar.color_picker("Distance borders")
        st.sidebar.subheader("Distance colormap")
        cmap_plot = st.sidebar.empty()
        hex_list = None
        proportion_list = None
        distance_min = st.sidebar.number_input("Min value:", value=-3)
        distance_max = st.sidebar.number_input("Max value:", value=3)
        if hex_list is None:
            hex_list = DEFAULT_HEX_LIST
            proportion_list = DEFAULT_PROPORTION_LIST
        else:
            hex_list = hex_list.split(",")
        st.sidebar.write("Distance scatter plot colormap:")
        for i in range(len(hex_list)):
            hex_list[i] = st.sidebar.color_picker(f"Color {i}", value=hex_list[i])
            proportion_list[i] = st.sidebar.number_input(
                f"Proportion {i}", value=proportion_list[i]
            )
        distance_cmap = get_distance_cmap(hex_list, proportion_list)
        cmap_plot.pyplot(get_colormap_plot(distance_cmap, distance_min, distance_max))
    else:
        distance_cmap = None
        distance_min = -3
        distance_max = 3

    distance_means = None
    distance_stds = None
    if df_distance is not None:
        df_distance = df_distance.loc[
            (df_distance.index >= min_time) & (df_distance.index <= max_time)
        ]
        min_distance_points = df_distance.apply(lambda x: x.count(), axis=1).min()
        if is_plot_distance_means:
            distance_means, distance_stds = get_means_error(
                df_distance, distance_error_type
            )

    if df_length is not None:
        df_length = df_length.loc[
            (df_length.index >= min_time) & (df_length.index <= max_time)
        ]
        min_length_points = df_length.apply(lambda x: x.count(), axis=1).min()
        if is_half_length:
            df_length = df_length.applymap(half)
        length_means, length_stds = get_means_error(df_length, length_error_type)

    else:
        length_means = None
        length_stds = None

    # st.header("Plot")

    plt = create_plot(
        df_distance,
        distance_means,
        distance_stds,
        distance_color,
        distance_size,
        distance_border,
        d_means_size,
        d_error_size,
        df_length,
        is_half_length,
        length_means,
        length_stds,
        length_color,
        l_means_size,
        l_error_size,
        distance_cmap,
        distance_min,
        distance_max,
        plot_width,
        plot_height,
        show_colorbar,
        show_grid,
        (ylim_min, ylim_max),
        title,
        xlabel,
        ylabel,
        title_size,
        xlabel_size,
        ylabel_size,
        xticks_size,
        yticks_size,
        marker,
    )

    outfile = st.text_input(
        "Save as:", value=os.path.join(os.path.abspath(os.getcwd()), "plot.eps")
    )
    allowed_formats = [".jpg", ".png", ".eps"]
    if st.button("Save"):
        if outfile[-4:] in allowed_formats:
            plt.savefig(outfile)
            st.info(f"Plot saved to {outfile}")
        else:
            st.error(
                f"Plot not saved: save file name can only ends with either '.jpg', '.png' or '.eps'"
            )

    if df_distance is not None:
        st.write(
            f"Minimum distance data points at each timestamp: {min_distance_points}"
        )
    if df_length is not None:
        st.write(f"Minimum length data points at each timestamp: {min_length_points}")

    st.pyplot(plt)
