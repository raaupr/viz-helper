import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xlrd
import yaml
from packaging import version

from ellipsoid import  plot_ellipsoid, get_ellipsoid_volume, get_min_vol_ellipse
from util_viz import LINE_STYLES, select_config

VERSION = "0.0.2"

cur_ver = version.parse(VERSION)

st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(
    page_title="Ellipsoid Viewer",
    page_icon=":pig_nose:",
    layout="centered",
    initial_sidebar_state="expanded",
)


def upload_data(container):
    fin = container.file_uploader(
        "Select the data file",
        type="xlsx",
        accept_multiple_files=False,
    )
    data_complete = False
    if fin:
        xls = pd.ExcelFile(xlrd.open_workbook(file_contents=fin.read(), on_demand=True))
        sheet = container.selectbox("Select data sheet:", xls.sheet_names)
        df = pd.read_excel(fin, sheet)
        row_start = 1
        df = pd.read_excel(fin, sheet, header=row_start)
        col_options = [f"{i}: {col}" for (i, col) in enumerate(df.columns)]
        float_col_options = [
            x
            for (i, x) in enumerate(col_options)
            if (df.dtypes[i] == "float64" or df.dtypes[i] == "int64")
        ]
        if float_col_options:
            col1, col2 = container.beta_columns(2)
            timestamp = col1.selectbox("Column timestamp:", float_col_options)
            name = col2.selectbox("Column names (opt):", ["None"] + col_options)
            cols = [
                0,
                1,
                2,
                int(timestamp.split(":")[0]),
            ]
            if name != "None":
                cols += [int(name.split(":")[0])]
            df = df.iloc[:, cols]
            df = df.dropna()
            data_complete = True
        else:
            container.warning(
                "Cannot find columns containing only numbers (required for x, y, z coordinates)"
            )
    else:
        container.warning("Please upload data file.")
        st.stop()
    return df, data_complete


# ================= PAGE STARTS HERE


st.title("Ellipsoid Viewer")
st.sidebar.markdown(
    f"[![v.{cur_ver}](https://img.shields.io/badge/version-{cur_ver}-green)](https://github.com/raaupr/viz-helper) "
)

st.sidebar.title("CONFIGURATION")
# st.sidebar.header("Upload data")

with st.sidebar.beta_expander("Upload data", expanded=True):
    df, data_complete = upload_data(st)

with st.beta_expander("Data", expanded=True):
    st.write(df)

if data_complete:
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    z = df.iloc[:, 2]
    times = df.iloc[:, 3]
    names = None
    if len(df.columns) > 4:
        names = df.iloc[:, 4]
    uniq_times = times.unique()

    with st.spinner("Computing alpha shapes..."):
        try:
            # -- compute alpha shapes
            volumes = []
            ellipsoids = {}
            for time in uniq_times:
                P = np.stack([
                    x[times == time],
                    y[times == time],
                    z[times == time]]).transpose()
                (center, radii, rotation) = get_min_vol_ellipse(P, .01)
                volume = get_ellipsoid_volume(radii)
                volumes.append(volume)
                ellipsoids[time] = {
                    "pos": P,
                    "center": center, 
                    "radii": radii, 
                    "rotation": rotation,
                    "volume": volume}
        except Exception as e:
            st.error("Unable to create ellipsoid from data, please check your data")
            print(e)
            st.stop()

    with st.beta_expander("Visualization", expanded=True):
        col1, col2 = st.beta_columns([1, 3])
        time = col1.selectbox("Choose timestamp:", uniq_times)
        col1.write(pd.DataFrame(ellipsoids[time]["pos"], columns=["x", "y", "z"]))
        col1.write(f"Volume: {ellipsoids[time]['volume']}")
        with st.spinner("Creating plot..."):
            # -- plot alpha shape
            fig, (minx, miny, minz, maxx, maxy, maxz) = plot_ellipsoid(
                ellipsoids[time]["pos"],
                ellipsoids[time]["center"],
                ellipsoids[time]["radii"],
                ellipsoids[time]["rotation"],
                P_text=names[times == time],
                plot_axes=True,
                cage_color="green",
                surface_color="blue",
                marker_color="red",
                axes_color="black",
            )
            st.write("Range: ")
            colx, coly, colz = st.beta_columns(3)
            x_range = colx.number_input("x-axis", value=maxx - minx)
            y_range = coly.number_input("y-axis", value=maxy - miny)
            z_range = colz.number_input("z-axis", value=maxz - minz)
            x_diff = (x_range - (maxx - minx)) / 2
            minx -= x_diff
            maxx += x_diff
            y_diff = (y_range - (maxy - miny)) / 2
            miny -= y_diff
            maxy += y_diff
            z_diff = (z_range - (maxz - minz)) / 2
            minz -= z_diff
            maxz += z_diff
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[minx, maxx]),
                    yaxis=dict(range=[miny, maxy]),
                    zaxis=dict(range=[minz, maxz])
                )
            )
            col2.plotly_chart(fig)

    with st.sidebar.beta_expander("Edit volume plot", expanded=False):
        # -- load config
        config = select_config(
            st,
            "upload_config",
        )
        if config is None:
            with open("viz_ellipsoid.yml") as fin_config:
                config = yaml.load(fin_config, Loader=yaml.Loader)
        # -- options
        st.subheader("Figure")
        config["fig_width"] = st.number_input(
            "Width:", min_value=0.0, value=config["fig_width"]
        )
        config["fig_height"] = st.number_input(
            "Height:", min_value=0.0, value=config["fig_height"]
        )
        st.subheader("Line")
        line_style_idx = LINE_STYLES.index(config["line_style"])
        config["line_style"] = st.selectbox(
            "Line style:", LINE_STYLES, index=line_style_idx
        )
        config["line_width"] = st.number_input(
            "Line width:", min_value=0.0, value=config["line_width"]
        )
        config["line_color"] = st.color_picker(
            "Line color:", value=config["line_color"]
        )
        st.subheader("Marker")
        config["marker"] = st.text_input("Marker type", value=config["marker"])
        config["marker_size"] = st.number_input(
            "Marker size:", min_value=0.0, value=config["marker_size"]
        )
        st.markdown(
            "<div style=' font-size: small;'>See <a href='https://matplotlib.org/api/markers_api.html'>here</a> for possible marker types</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.beta_columns(2)
        config["marker_facecolor"] = col1.color_picker(
            "Marker face color:", value=config["marker_facecolor"]
        )
        config["marker_edgecolor"] = col2.color_picker(
            "Marker edge color", value=config["marker_edgecolor"]
        )
        st.subheader("Title")
        config["title"] = st.text_input("Title:", value=config["title"])
        config["title_size"] = st.number_input(
            "Title size:", min_value=0.0, value=config["title_size"]
        )
        config["title_color"] = st.color_picker(
            "Title color:", value=config["title_color"]
        )
        st.subheader("x-axis")
        config["xlim_min"] = st.number_input(
            "Min value: ", key="xlim_min", value=config["xlim_min"]
        )
        config["xlim_max"] = st.number_input(
            "Max value: ", key="xlim_max", value=config["xlim_max"]
        )
        config["xlabel"] = st.text_input("Label:", key="xlabel", value=config["xlabel"])
        config["xlabel_size"] = st.number_input(
            "Label size:", key="xlabel_size", min_value=0.0, value=config["xlabel_size"]
        )
        config["xticks_size"] = st.number_input(
            "Ticks size:", key="xticks_size", min_value=0.0, value=config["xticks_size"]
        )
        config["xticks_interval"] = st.number_input(
            "Ticks interval:",
            key="xticks_interval",
            min_value=0.0,
            value=config["xticks_interval"],
        )
        col1, col2 = st.beta_columns(2)
        config["xlabel_color"] = col1.color_picker(
            "Label color:", key="xlabel_color", value=config["xlabel_color"]
        )
        config["xticks_color"] = col2.color_picker(
            "Ticks color:", key="xticks_color", value=config["xticks_color"]
        )
        st.subheader("y-axis")
        config["ylim_min"] = st.number_input(
            "Min value: ", key="ylim_min", value=config["ylim_min"]
        )
        config["ylim_max"] = st.number_input(
            "Max value: ", key="ylim_max", value=config["ylim_max"]
        )
        config["ylabel"] = st.text_input("Label:", key="ylabel", value=config["ylabel"])
        config["ylabel_size"] = st.number_input(
            "Label size:", key="ylabel_size", min_value=0.0, value=config["ylabel_size"]
        )
        config["yticks_size"] = st.number_input(
            "Ticks size:", key="yticks_size", min_value=0.0, value=config["yticks_size"]
        )
        config["yticks_interval"] = st.number_input(
            "Ticks interval:",
            key="yticks_interval",
            min_value=0.0,
            value=config["yticks_interval"],
        )
        col1, col2 = st.beta_columns(2)
        config["ylabel_color"] = col1.color_picker(
            "Label color:", key="ylabel_color", value=config["ylabel_color"]
        )
        config["yticks_color"] = col2.color_picker(
            "Ticks color:", key="yticks_color", value=config["yticks_color"]
        )
        st.subheader("Elements")
        config["show_grid"] = st.checkbox("Show grid", value=config["show_grid"])
        st.subheader("Save configurations")
        save_config_file = st.text_input(
            "Save configurations as:",
            value=os.path.join(os.path.abspath(os.getcwd()), "config.yml"),
        )
        if st.button("Save", key="save_config"):
            with open(save_config_file, "w") as fout:
                fout.write(yaml.dump(config, Dumper=yaml.Dumper))
            st.info(f"Config saved to {save_config_file}")

    with st.beta_expander("Volume Plot", expanded=True):
        # -- plot
        fig, ax = plt.subplots(figsize=(config["fig_width"], config["fig_height"]))
        # -- line
        ax.plot(
            uniq_times,
            volumes,
            marker=config["marker"],
            linestyle=config["line_style"],
            linewidth=config["line_width"],
            markersize=config["marker_size"],
            color=config["line_color"],
            markerfacecolor=config["marker_facecolor"],
            markeredgecolor=config["marker_edgecolor"],
        )
        # -- title
        ax.set_title(
            config["title"],
            {"fontsize": config["title_size"], "color": config["title_color"]},
        )
        # -- x-axis
        ax.set_xlabel(
            config["xlabel"], size=config["xlabel_size"], color=config["xlabel_color"]
        )
        ax.set_xlim((config["xlim_min"], config["xlim_max"]))
        # if names is None:
        ax.set_xticks(
            np.arange(
                config["xlim_min"],
                config["xlim_max"] + config["xticks_interval"],
                config["xticks_interval"],
            )
        )
        ax.tick_params(
            axis="x", colors=config["xticks_color"], labelsize=config["xticks_size"]
        )
        # -- y-axis
        ax.set_ylabel(
            config["ylabel"], size=config["ylabel_size"], color=config["ylabel_color"]
        )
        ax.set_ylim((config["ylim_min"], config["ylim_max"]))
        ax.set_yticks(
            np.arange(
                config["ylim_min"],
                config["ylim_max"] + config["yticks_interval"],
                config["yticks_interval"],
            )
        )
        ax.tick_params(
            axis="y", colors=config["yticks_color"], labelsize=config["yticks_size"]
        )
        if config["show_grid"]:
            plt.grid()

        col1, col2 = st.beta_columns([1, 3])
        # -- show volumes
        df_vol = pd.DataFrame({"timestamp": uniq_times, "volume": volumes})
        col1.write(df_vol)
        # -- save volumes
        vol_outfile = col1.text_input(
            "Save volumes as:",
            value=os.path.join(os.path.abspath(os.getcwd()), "volumes.csv"),
        )
        if col1.button("Save volumes"):
            with st.spinner("Saving volumes..."):
                df_vol.to_csv(vol_outfile)
            col1.info(f"Volumes saved to {vol_outfile}")
        # -- show plot
        col2.pyplot(plt)
        # -- save plot
        outfile = col2.text_input(
            "Save plot as:",
            value=os.path.join(os.path.abspath(os.getcwd()), "plot.eps"),
        )
        if col2.button("Save plot"):
            if (
                outfile[-4:] in plt.gcf().canvas.get_supported_filetypes().keys()
                or outfile[-3:] in plt.gcf().canvas.get_supported_filetypes().keys()
            ):
                with st.spinner("Saving plot..."):
                    plt.savefig(outfile)
                col2.info(f"Plot saved to {outfile}")
            else:
                col2.error(
                    f"Filename should end with one of the following extensions: {list(plt.gcf().canvas.get_supported_filetypes().keys())}"
                )
