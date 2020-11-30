"""GUI for plot_distance_length.py"""

import os

import streamlit as st
import yaml

from plot_distance_length import (
    create_plot,
    get_means_error,
    half,
    read_file,
)
from util_colormap import get_colormap_plot, get_continuous_cmap_bypoint

ERROR_BAR_TYPES = ["None", "SD", "SEM"]
ALLOWED_OUTFILE_EXT = [".jpg", ".png", ".eps"]

st.title("Visualize Chromosome Distance & Splindle Length vs Time")
st.write("v.30112020")

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("Customization options")

#### options
st.sidebar.header("Load configuration (optional)")
with open("viz.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)
fin_config = st.sidebar.file_uploader(
    "Select configuration file (optional) (.yml file only!)", type="yml"
)
if fin_config:
    config = yaml.load(fin_config, Loader=yaml.Loader)
    st.info("Configuration file loaded")

# -- Get data
st.sidebar.header("Distance")
df_distance = None

config["DISTANCE"]["plot_distance"] = st.sidebar.checkbox("Plot distance", value=config["DISTANCE"]["plot_distance"])
if config["DISTANCE"]["plot_distance"]:
    fin_distance = st.sidebar.file_uploader(
        "Select the chromosome-spindle distance file (.xlsx file only!)", type="xlsx"
    )
    if fin_distance:
        df_distance = read_file(fin_distance)
        st.info(
            f"Reading distance file successful: found {len(df_distance.columns)} data points."
        )
        config["DISTANCE"]["plot_distance_means"] = st.sidebar.checkbox(
            "Plot distance means", value=config["DISTANCE"]["plot_distance_means"]
        )
        config["DISTANCE"]["means_error_type"] = st.sidebar.selectbox(
            "Error bar type:", ERROR_BAR_TYPES, key="d_error_type", index=ERROR_BAR_TYPES.index(config["DISTANCE"]["means_error_type"])
        )
    else:
        st.warning("\u2190 Please select the distance dataset file.")
        st.sidebar.warning("Please select the distance dataset file.")
st.sidebar.header("Length")
df_length = None
config["LENGTH"]["plot_length_means"] = st.sidebar.checkbox("Plot length", value=config["LENGTH"]["plot_length_means"])
if config["LENGTH"]["plot_length_means"]:
    fin_length = st.sidebar.file_uploader(
        "Select the spindle length file (.xlsx file only!)", type="xlsx"
    )
    if fin_length:
        df_length = read_file(fin_length)
        st.info(
            f"Reading length file successful: found {len(df_length.columns)} data points."
        )
        config["LENGTH"]["half_symmetric_length"] = st.sidebar.checkbox(
            "Make length plot symmetrical w.r.t. the center", value=config["LENGTH"]["half_symmetric_length"]
        )
        config["LENGTH"]["means_error_type"] = st.sidebar.selectbox(
            "Error bar type:", ERROR_BAR_TYPES, key="l_error_type", index=ERROR_BAR_TYPES.index(config["LENGTH"]["means_error_type"])
        )
    else:
        st.warning("\u2190 Please select the length dataset file.")
        st.sidebar.warning("Please select the length dataset file.")
if not config["DISTANCE"]["plot_distance"] and not config["LENGTH"]["plot_length_means"]:
    st.write("\u2190 Please select what to plot.")


if (config["DISTANCE"]["plot_distance"] and df_distance is not None) or (
    config["LENGTH"]["plot_length_means"] and df_length is not None
):
    # ---- CUSTOMIZATION
    st.sidebar.header("Ranges:")
    st.sidebar.subheader("x-axis")
    if df_distance is not None:
        min_default = min(df_distance.index)
        max_default = max(df_distance.index)
        if df_length is not None:
            min_default = min(min_default, min(df_length.index))
            max_default = max(max_default, max(df_length.index))
    if df_length is not None:
        min_default = min(df_length.index)
        max_default = max(df_length.index)
            
    xlim_min = config["PLOT"]["xlim_min"]
    xlim_max = config["PLOT"]["xlim_max"]
    if  xlim_min > min_default and xlim_min < max_default:
        min_default = xlim_min
    if xlim_max > min_default and xlim_max < max_default:
        max_default = xlim_max
    config["PLOT"]["xlim_min"] = st.sidebar.number_input(
        "Min time:", min_value=min_default, max_value=max_default, value=min_default
    )
    config["PLOT"]["xlim_max"] = st.sidebar.number_input(
        "Max time:", min_value=min_default, max_value=max_default, value=max_default
    )
    st.sidebar.subheader("y-axis")
    config["PLOT"]["ylim_min"] = st.sidebar.number_input("Min value:", value=config["PLOT"]["ylim_min"])
    config["PLOT"]["ylim_max"] = st.sidebar.number_input("Max value:", value=config["PLOT"]["ylim_max"])

    st.sidebar.header("Texts:")
    config["PLOT"]["title"] = st.sidebar.text_input("Title:", value=config["PLOT"]["title"])
    config["PLOT"]["xlabel"] = st.sidebar.text_input(
        "x-axis label:", value=config["PLOT"]["xlabel"]
    )
    config["PLOT"]["ylabel"] = st.sidebar.text_input(
        "y-axis label:", value=config["PLOT"]["ylabel"]
    )

    st.sidebar.header("Sizes:")
    config["PLOT"]["title_size"] = st.sidebar.number_input("Title size:", value=config["PLOT"]["title_size"], min_value=0.0)
    config["PLOT"]["xlabel_size"] = st.sidebar.number_input("x-axis label size:", value=config["PLOT"]["xlabel_size"], min_value=0.0)
    config["PLOT"]["xticks_size"] = st.sidebar.number_input("x-axis tick labels size:", value=config["PLOT"]["xticks_size"], min_value=0.0)
    config["PLOT"]["ylabel_size"] = st.sidebar.number_input("y-axis label size:", value=config["PLOT"]["ylabel_size"], min_value=0.0)
    config["PLOT"]["yticks_size"] = st.sidebar.number_input("y-axis tick labels size:", value=config["PLOT"]["yticks_size"], min_value=0.0)
    config["PLOT"]["width"] = st.sidebar.number_input("Plot width:", value=config["PLOT"]["width"], min_value=0.0)
    config["PLOT"]["height"] = st.sidebar.number_input("Plot height:", value=config["PLOT"]["height"], min_value=0.0)
    
    if df_distance is not None:
        config["DISTANCE"]["point_size"] = st.sidebar.number_input("Size of distance points:", value=config["DISTANCE"]["point_size"], min_value=0.0)
    if config["DISTANCE"]["plot_distance_means"]:
        config["DISTANCE"]["means_size"] = st.sidebar.number_input("Distance means line width:", value=config["DISTANCE"]["means_size"], min_value=0.0)
        if config["DISTANCE"]["means_error_type"] != "None":
            config["DISTANCE"]["error_size"] = st.sidebar.number_input("Distance error line width:", value=config["DISTANCE"]["error_size"], min_value=0.0)
    if config["LENGTH"]["plot_length_means"]:
        config["LENGTH"]["means_size"] = st.sidebar.number_input("Length means line width:", value=config["LENGTH"]["means_size"], min_value=0.0)
        if config["LENGTH"]["means_error_type"] != "None":
            config["LENGTH"]["error_size"] = st.sidebar.number_input("Length error line width:", value=config["LENGTH"]["error_size"], min_value=0.0)

    st.sidebar.header("Elements:")
    if df_distance is not None:
        config["DISTANCE"]["point_type"] = st.sidebar.text_input("Marker type for distance points:", value=config["DISTANCE"]["point_type"])
        st.sidebar.markdown(
            "See [here](https://matplotlib.org/api/markers_api.html) for possible marker types"
        )
    if df_distance is not None:
        config["DISTANCE"]["show_colorbar"] = st.sidebar.checkbox("Show color bar", value=config["DISTANCE"]["show_colorbar"])
    config["PLOT"]["show_grid"] = st.sidebar.checkbox("Show grid", value = config["PLOT"]["show_grid"])

    st.sidebar.header("Colors:")
    if df_distance is not None and config["DISTANCE"]["plot_distance_means"]:
        config["DISTANCE"]["means_color"] = st.sidebar.color_picker(
            "Distance means:", value=config["DISTANCE"]["means_color"]
        )
        config["DISTANCE"]["error_color"] = st.sidebar.color_picker(
            "Distance means error:", value=config["DISTANCE"]["error_color"]
        )
    if df_length is not None:
        config["LENGTH"]["means_color"] = st.sidebar.color_picker("Length means:", value=config["LENGTH"]["means_color"])
        config["LENGTH"]["error_color"] = st.sidebar.color_picker("Length means error:", value=config["LENGTH"]["error_color"])
    if df_distance is not None:
        config["DISTANCE"]["point_use_border"] = st.sidebar.checkbox(
            "Use border on distance marker points", value = config["DISTANCE"]["point_use_border"]
        )
        if config["DISTANCE"]["point_use_border"]:
            if config["DISTANCE"]["point_border_color"] != "face":
                config["DISTANCE"]["point_border_color"] = st.sidebar.color_picker("Distance borders:", value = config["DISTANCE"]["point_border_color"])
            else:
                config["DISTANCE"]["point_border_color"] = st.sidebar.color_picker("Distance borders:")
        if not config["DISTANCE"]["point_use_border"]:
            config["DISTANCE"]["point_border_color"] = "face"

        st.sidebar.subheader("Distance colormap")
        cmap_plot = st.sidebar.empty()
        config["DISTANCE"]["colormap_min"] = st.sidebar.number_input("Min value:", value=config["DISTANCE"]["colormap_min"])
        config["DISTANCE"]["colormap_max"] = st.sidebar.number_input("Max value:", value=config["DISTANCE"]["colormap_max"], min_value = config["DISTANCE"]["colormap_min"] + 0.1)

        st.sidebar.write("Distance scatter plot colormap:")
        nb_colormap = st.sidebar.number_input("Number of colors:", value=len(config["DISTANCE"]["colormap_hex"]))
        st.sidebar.write("Colors: (point values should be in increasing order)")
        new_d_colormap_hex = []
        new_d_colormap_float = []
        for i in range(nb_colormap):
            if i < len(config["DISTANCE"]["colormap_hex"]):
                hex_val = config["DISTANCE"]["colormap_hex"][i]
                hex = st.sidebar.color_picker(f"Color {i}:", value=hex_val)
                if i == 0:
                    val = st.sidebar.number_input(
                        f"Point value {i}:", 
                        value=config["DISTANCE"]["colormap_min"], 
                        min_value=config["DISTANCE"]["colormap_min"], 
                        max_value=config["DISTANCE"]["colormap_min"]
                    )    
                elif i == nb_colormap - 1:
                    val = st.sidebar.number_input(
                        f"Point value {i}:", 
                        value=config["DISTANCE"]["colormap_max"], 
                        min_value=config["DISTANCE"]["colormap_max"], 
                        max_value=config["DISTANCE"]["colormap_max"]
                    )
                else:
                    float_val = max(
                        max(config["DISTANCE"]["colormap_min"], config["DISTANCE"]["colormap_float"][i]), 
                        min(config["DISTANCE"]["colormap_max"], config["DISTANCE"]["colormap_float"][i]))
                    val = st.sidebar.number_input(
                        f"Point value {i}:", 
                        value=float_val, 
                        min_value=config["DISTANCE"]["colormap_min"], 
                        max_value=config["DISTANCE"]["colormap_max"]
                    )
            else:
                hex = st.sidebar.color_picker(f"Color {i}:")
                if i == 0:
                    val = st.sidebar.number_input(
                        f"Point value {i}:", 
                        value=config["DISTANCE"]["colormap_min"], 
                        min_value=config["DISTANCE"]["colormap_min"], 
                        max_value=config["DISTANCE"]["colormap_min"]
                    )    
                elif i == nb_colormap - 1:
                    val = st.sidebar.number_input(
                        f"Point value {i}:", 
                        value=config["DISTANCE"]["colormap_max"], 
                        min_value=config["DISTANCE"]["colormap_max"], 
                        max_value=config["DISTANCE"]["colormap_max"]
                    )
                else:
                    val = st.sidebar.number_input(
                        f"Point value {i}:", 
                        value=config["DISTANCE"]["colormap_max"], 
                        min_value=config["DISTANCE"]["colormap_min"],
                        max_value=config["DISTANCE"]["colormap_max"])
            new_d_colormap_hex.append(hex)
            new_d_colormap_float.append(val)
        config["DISTANCE"]["colormap_hex"] = new_d_colormap_hex
        config["DISTANCE"]["colormap_float"] = new_d_colormap_float
        distance_cmap = get_continuous_cmap_bypoint(config["DISTANCE"]["colormap_hex"], config["DISTANCE"]["colormap_float"])
        cmap_plot.pyplot(get_colormap_plot(distance_cmap, config["DISTANCE"]["colormap_min"], config["DISTANCE"]["colormap_max"]))

    distance_means = None
    distance_stds = None
    if df_distance is not None:
        df_distance = df_distance.loc[
            (df_distance.index >= config["PLOT"]["xlim_min"]) & (df_distance.index <= config["PLOT"]["xlim_max"])
        ]
        min_distance_points = df_distance.apply(lambda x: x.count(), axis=1).min()
        if config["DISTANCE"]["plot_distance_means"]:
            distance_means, distance_stds = get_means_error(
                df_distance, config["DISTANCE"]["means_error_type"]
            )

    if df_length is not None:
        df_length = df_length.loc[
            (df_length.index >= config["PLOT"]["xlim_min"]) & (df_length.index <= config["PLOT"]["xlim_max"])
        ]
        min_length_points = df_length.apply(lambda x: x.count(), axis=1).min()
        if config["LENGTH"]["half_symmetric_length"]:
            df_length = df_length.applymap(half)
        length_means, length_stds = get_means_error(df_length, config["LENGTH"]["means_error_type"])

    else:
        length_means = None
        length_stds = None

    # st.header("Plot")

    with st.spinner("Creating plot..."):
        plt = create_plot(
            df_distance,
            distance_means,
            distance_stds,
            config["DISTANCE"]["means_color"],
            config["DISTANCE"]["error_color"],
            config["DISTANCE"]["point_size"],
            config["DISTANCE"]["point_border_color"],
            config["DISTANCE"]["means_size"],
            config["DISTANCE"]["error_size"],
            df_length,
            config["LENGTH"]["half_symmetric_length"],
            length_means,
            length_stds,
            config["LENGTH"]["means_color"],
            config["LENGTH"]["error_color"],
            config["LENGTH"]["means_size"],
            config["LENGTH"]["error_size"],
            distance_cmap,
            config["DISTANCE"]["colormap_min"],
            config["DISTANCE"]["colormap_max"],
            config["PLOT"]["width"],
            config["PLOT"]["height"],
            config["DISTANCE"]["show_colorbar"],
            config["PLOT"]["show_grid"],
            (config["PLOT"]["ylim_min"], config["PLOT"]["ylim_max"]),
            config["PLOT"]["title"],
            config["PLOT"]["xlabel"],
            config["PLOT"]["ylabel"],
            config["PLOT"]["title_size"],
            config["PLOT"]["xlabel_size"],
            config["PLOT"]["ylabel_size"],
            config["PLOT"]["xticks_size"],
            config["PLOT"]["yticks_size"],
            config["DISTANCE"]["point_type"],
        )

    config["PLOT"]["save_file"] = st.text_input(
        "Save as:", value=config["PLOT"]["save_file"]
    )
    
    if st.button("Save"):
        if config["PLOT"]["save_file"][-4:] in ALLOWED_OUTFILE_EXT:
            plt.savefig(config["PLOT"]["save_file"])
            st.info(f"Plot saved to {config['PLOT']['save_file']}")
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


st.sidebar.header("Save configurations")
save_config_file = st.sidebar.text_input(
    "Save configurations as:", value="config.yml"
)
if st.sidebar.button("Save", key="save_config"):
    with open(save_config_file, "w") as fout:
        fout.write(yaml.dump(config, Dumper=yaml.Dumper))
    st.sidebar.info(f"Config saved to {save_config_file}")