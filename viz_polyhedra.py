import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xlrd
import yaml
from packaging import version

from polyhedron import alpha_shape_3d_autoalpha, compute_volume
from util_viz import LINE_STYLES, select_config, LEGEND_LOCATIONS

VERSION = "0.0.0"

COLOR_ROW = "#FAEDCB"
COLOR_FRAME = "#F7D9C4"
COLOR_X = "#C9E4DE"
COLOR_Y = "#C6DEF1"
COLOR_Z = "#DBCDF0"

IDX_TIME, IDX_FRAME, IDX_X, IDX_Y, IDX_Z = [0, 1, 2, 3, 4]

cur_ver = version.parse(VERSION)

st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(
    page_title="Polyhedra",
    page_icon=":shark:",
    layout="centered",
    initial_sidebar_state="expanded",
)


def upload_files(container):
    fin_list = container.file_uploader(
        "Select the data file(s) (can be multiple)",
        type="xlsx",
        accept_multiple_files=True,
    )
    return fin_list


def highlight_rows(val):
    return f"background-color: {COLOR_ROW}"


def highlight_frame(val):
    return f"background-color: {COLOR_FRAME}"


def highlight_x(val):
    return f"background-color: {COLOR_X}"


def highlight_y(val):
    return f"background-color: {COLOR_Y}"


def highlight_z(val):
    return f"background-color: {COLOR_Z}"


def edit_data(container, fin_list):
    dfs_ori = []
    dfs = []
    dfs_status = []
    dfs_styler = []
    data_names = []
    if fin_list:
        for i, fin in enumerate(fin_list):
            status = False
            with container.beta_expander(f"File {i+1}"):
                xls = pd.ExcelFile(
                    xlrd.open_workbook(file_contents=fin.read(), on_demand=True)
                )
                name = st.text_input("Data name:", value=f"Data {i+1}", key=f"name{i}")
                sheet = st.selectbox(
                    "Select data sheet:", xls.sheet_names, key=f"sheet{i}"
                )
                df_ori = pd.read_excel(fin, sheet)
                row_start = st.number_input(
                    "Start of data row:",
                    min_value=0,
                    max_value=len(df_ori) - 1,
                    value=0,
                    key=f"row_start{i}",
                )
                df_ori = df_ori.style.applymap(
                    highlight_rows, subset=pd.IndexSlice[row_start:, :]
                )
                df = pd.read_excel(fin, sheet, header=row_start)
                col_options = [f"{i}: {col}" for (i, col) in enumerate(df.columns)]
                float_col_options = [
                    x for (i, x) in enumerate(col_options) if df.dtypes[i] == "float64"
                ]
                if float_col_options:
                    x_col = st.selectbox("Column x:", float_col_options, key=f"xcol{i}")
                    x_idx = int(x_col.split(":")[0])
                    df_ori.applymap(
                        highlight_x,
                        subset=pd.IndexSlice[row_start:, [df_ori.columns[x_idx]]],
                    )
                    y_col = st.selectbox("Column y:", float_col_options, key=f"ycol{i}")
                    y_idx = int(y_col.split(":")[0])
                    df_ori.applymap(
                        highlight_y,
                        subset=pd.IndexSlice[row_start:, [df_ori.columns[y_idx]]],
                    )
                    z_col = st.selectbox("Column z:", float_col_options, key=f"zcol{i}")
                    z_idx = int(z_col.split(":")[0])
                    df_ori.applymap(
                        highlight_z,
                        subset=pd.IndexSlice[row_start:, [df_ori.columns[z_idx]]],
                    )
                    frame_col = st.selectbox(
                        "Column frame:", col_options, key=f"frame{i}"
                    )
                    frame_idx = int(frame_col.split(":")[0])
                    df_ori.applymap(
                        highlight_frame,
                        subset=pd.IndexSlice[row_start:, [df_ori.columns[frame_idx]]],
                    )
                    # -- frame 0
                    frame_0 = st.selectbox(
                        "Frame 0:", df.iloc[:, frame_idx].unique(), key=f"frame0{i}"
                    )
                    df["time alignment"] = [
                        (10 * (i - frame_0)) for i in df.iloc[:, frame_idx]
                    ]
                    cols = [len(df.columns) - 1, frame_idx, x_idx, y_idx, z_idx]
                    df = df.iloc[:, cols]
                    # -- style
                    col_names = [
                        "time alignment",
                        f"frame: {df.columns[1]}",
                        f"x: {df.columns[2]}",
                        f"y: {df.columns[3]}",
                        f"z: {df.columns[4]}",
                    ]
                    df.columns = col_names
                    df_styler = df.style.applymap(
                        highlight_frame, subset=pd.IndexSlice[:, [col_names[1]]]
                    )
                    df_styler.applymap(
                        highlight_x, subset=pd.IndexSlice[:, [col_names[2]]]
                    )
                    df_styler.applymap(
                        highlight_y, subset=pd.IndexSlice[:, [col_names[3]]]
                    )
                    df_styler.applymap(
                        highlight_z, subset=pd.IndexSlice[:, [col_names[4]]]
                    )
                    dfs_styler.append(df_styler)
                    status = True
                else:
                    st.error(
                        f"File {i+1} error: cannot find columns containing only numbers (required for x, y, z coordinates). You might need to change to the appropriate sheet, or change the row start of the data."
                    )
                    dfs_styler.append(None)
                dfs_ori.append(df_ori)
                dfs.append(df)
                dfs_status.append(status)
                data_names.append(name)
    else:
        container.warning("Please upload data file(s).")
        st.stop()
    return dfs_ori, dfs, dfs_status, dfs_styler, data_names


st.title("Polyhedra Volumes Plot")
st.sidebar.markdown(
    f"[![v.{cur_ver}](https://img.shields.io/badge/version-{cur_ver}-green)](https://github.com/raaupr/viz-helper) "
)

st.sidebar.title("CONFIGURATION")

with st.sidebar.beta_expander("Upload data", expanded=True):
    fin_list = upload_files(st)

with st.spinner("Processing data..."):
    dfs_ori, dfs, dfs_status, dfs_styler, data_names = edit_data(st.sidebar, fin_list)

st.header("Data")
if len(dfs) > 0:
    st.write(
        f"""
    <div style='width: 100%; text-align:center; display: flex'>
        <div style='text-align: left; display: inline; font-size: small; padding: 2px; margin-right: 1rem'>Legend:</div> 
        <div style='background: {COLOR_ROW}; text-align: center; border-radius: 5px; display: inline; font-size: small; padding: 2px; width: 100%; margin-right: 1rem'>data rows</div>
        <div style='background: {COLOR_FRAME}; text-align: center; border-radius: 5px; display: inline; font-size: small; padding: 2px; width: 100%; margin-right: 1rem'>frame data</div>
        <div style='background: {COLOR_X}; text-align: center; border-radius: 5px; display: inline; font-size: small; padding: 2px; width: 100%; margin-right: 1rem'>x data</div>
        <div style='background: {COLOR_Y}; text-align: center; border-radius: 5px; display: inline; font-size: small; padding: 2px; width: 100%; margin-right: 1rem'>y data</div>
        <div style='background: {COLOR_Z}; text-align: center; border-radius: 5px; display: inline; font-size: small; padding: 2px; width: 100%; margin-right: 1rem'>z data</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
for i, (df_ori, df, df_status, df_styler, data_name) in enumerate(
    zip(dfs_ori, dfs, dfs_status, dfs_styler, data_names)
):
    with st.beta_expander(f"File {i+1}: {data_name}", expanded=True):
        st.markdown("***Original data table:***")
        st.write(df_ori)
        if df_status:
            frames = df.iloc[:, 0]
            x = df.iloc[:, 1]
            y = df.iloc[:, 2]
            z = df.iloc[:, 3]
            st.markdown("***Selected data:***")
            st.write(df_styler)
        else:
            st.error(
                f"File {i+1} error: cannot find columns containing only numbers (required for x, y, z coordinates). You might need to change to the appropriate sheet, or change the row start of the data."
            )

# -- PLOT
st.header("Volume")
error_df_idxs = [i + 1 for (i, x) in enumerate(dfs_status) if not x]
if error_df_idxs:
    error_files = ", ".join([f"File {i}" for i in error_df_idxs])
    st.error(f"Cannot compute volumes, data error on file(s): {error_files}")
    st.stop()

with st.spinner("Computing volumes..."):
    df_volumes = []
    for df, data_name in zip(dfs, data_names):
        x = df.iloc[:, IDX_X]
        y = df.iloc[:, IDX_Y]
        z = df.iloc[:, IDX_Z]
        volumes = []
        try:
            # -- compute alpha shapes
            times = df.iloc[:, IDX_TIME]
            for time in times.unique():
                pos = np.array(
                    [
                        np.array(x[times == time]),
                        np.array(y[times == time]),
                        np.array(z[times == time]),
                    ]
                ).transpose()
                _, _, triangles = alpha_shape_3d_autoalpha(pos)
                volume = compute_volume(pos, triangles)
                volumes.append(volume)
        except Exception:
            st.error("Unable to create polyhedra from data, please check your data")
            st.stop()
        df_vol = pd.DataFrame({"time": times.unique(), data_name: volumes})
        df_vol = df_vol.set_index("time")
        df_volumes.append(df_vol)
        df_volumes_concat = pd.concat(df_volumes, axis=1, join="outer")

with st.beta_expander("Volume data"):              
    st.write(df_volumes_concat)
    vol_outfile = st.text_input("Save volumes as:", value=os.path.join(os.path.abspath(os.getcwd()), "volumes.csv"))
    if st.button("Save volumes"):
        with st.spinner("Saving volumes..."):
            df_vol.to_csv(vol_outfile)
        st.info(f"Volumes saved to {vol_outfile}")     

with st.sidebar.beta_expander("Edit volume plot", expanded=False):
    # -- load config
    config = select_config(
        st,
        "upload_config",
    )
    if config is None:
        with open("viz_polyhedra.yml") as fin_config:
            config = yaml.load(fin_config, Loader=yaml.Loader)
    # -- options
    st.subheader("Figure")
    col1, col2 = st.beta_columns(2)
    config["fig_width"] = col1.number_input(
        "Width:", min_value=0.0, value=config["fig_width"]
    )
    config["fig_height"] = col2.number_input(
        "Height:", min_value=0.0, value=config["fig_height"]
    )
    new_line_styles = []
    new_line_widths = []
    new_line_colors = []
    new_markers = []
    new_marker_sizes = []
    new_marker_facecolors = []
    new_marker_edgecolors = []
    for i in range(len(data_names)):
        if i < len(config["line_styles"]):
            new_line_styles.append(config["line_styles"][i])
            new_line_widths.append(config["line_widths"][i])
            new_line_colors.append(config["line_colors"][i])
            new_markers.append(config["markers"][i])
            new_marker_sizes.append(config["marker_sizes"][i])
            new_marker_facecolors.append(config["marker_facecolors"][i])
            new_marker_edgecolors.append(config["marker_edgecolors"][i])
        else:
            new_line_styles.append(config["line_styles"][0])
            new_line_widths.append(config["line_widths"][0])
            new_line_colors.append(config["line_colors"][0])
            new_markers.append(config["markers"][0])
            new_marker_sizes.append(config["marker_sizes"][0])
            new_marker_facecolors.append(config["marker_facecolors"][0])
            new_marker_edgecolors.append(config["marker_edgecolors"][0])
    config["line_styles"] = new_line_styles
    config["line_widths"] = new_line_widths
    config["line_colors"] = new_line_colors
    config["markers"] = new_markers
    config["marker_sizes"] = new_marker_sizes
    config["marker_facecolors"] = new_marker_facecolors
    config["marker_edgecolors"] = new_marker_edgecolors
    for i in range(len(data_names)):
        st.subheader(data_names[i])
        col1, col2 = st.beta_columns(2)
        line_style_idx = LINE_STYLES.index(config["line_styles"][i])
        config["line_styles"][i] = col1.selectbox(
            "Line style:", LINE_STYLES, index=line_style_idx,
            key = f"line_style{i}"
        )
        config["line_widths"][i] = col2.number_input(
            "Line width:", min_value=0.0, value=config["line_widths"][i],
            key = f"line_width{i}"
        )
        config["line_colors"][i] = st.color_picker(
            "Line color:", value=config["line_colors"][i],
            key = f"line_color{i}"
        )
        col1, col2 = st.beta_columns(2)
        config["markers"][i] = col1.text_input("Marker type", value=config["markers"][i], key=f"marker{i}")
        config["marker_sizes"][i] = col2.number_input(
            "Marker size:", min_value=0.0, value=config["marker_sizes"][i], key=f"marker_size{i}"
        )
        st.markdown(
            "<div style=' font-size: small;'>See <a href='https://matplotlib.org/api/markers_api.html'>here</a> for possible marker types</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.beta_columns(2)
        config["marker_facecolors"][i] = col1.color_picker(
            "Marker face color:", value=config["marker_facecolors"][i], key=f"marker_facecolor{i}"
        )
        config["marker_edgecolors"][i] = col2.color_picker(
            "Marker edge color", value=config["marker_edgecolors"][i], key=f"marker_edgecolor{i}"
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
    col1, col2 = st.beta_columns(2)
    config["xlim_min"] = col1.number_input(
        "Min value: ", key="xlim_min", value=config["xlim_min"]
    )
    config["xlim_max"] = col2.number_input(
        "Max value: ", key="xlim_max", value=config["xlim_max"]
    )
    config["xlabel"] = st.text_input("Label:", key="xlabel", value=config["xlabel"])
    config["xlabel_size"] = st.number_input(
        "Label size:", key="xlabel_size", min_value=0.0, value=config["xlabel_size"]
    )
    col1, col2 = st.beta_columns(2)
    config["xticks_size"] = col1.number_input(
        "Ticks size:", key="xticks_size", min_value=0.0, value=config["xticks_size"]
    )
    config["xticks_interval"] = col2.number_input(
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
    col1, col2 = st.beta_columns(2)
    config["ylim_min"] = col1.number_input(
        "Min value: ", key="ylim_min", value=config["ylim_min"]
    )
    config["ylim_max"] = col2.number_input(
        "Max value: ", key="ylim_max", value=config["ylim_max"]
    )
    config["ylabel"] = st.text_input("Label:", key="ylabel", value=config["ylabel"])
    config["ylabel_size"] = st.number_input(
        "Label size:", key="ylabel_size", min_value=0.0, value=config["ylabel_size"]
    )
    col1, col2 = st.beta_columns(2)
    config["yticks_size"] = col1.number_input(
        "Ticks size:", key="yticks_size", min_value=0.0, value=config["yticks_size"]
    )
    config["yticks_interval"] = col2.number_input(
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
    config["show_legend"] = st.checkbox("Show legend", value=config["show_legend"])
    if config["show_legend"]:
        config["legend_loc"] = st.selectbox(
            "Legend location:", 
            LEGEND_LOCATIONS,
            index = LEGEND_LOCATIONS.index(config["legend_loc"])
        )
        config["legend_size"] = st.number_input(
            "Legend font size:",
            min_value = 0.0,
            value = config["legend_size"]
        )
    st.subheader("Save configurations")
    save_config_file = st.text_input(
        "Save configurations as:", value=os.path.join(os.path.abspath(os.getcwd()), "config.yml")
    )
    if st.button("Save", key="save_config"):
        with open(save_config_file, "w") as fout:
            fout.write(yaml.dump(config, Dumper=yaml.Dumper))
        st.info(f"Config saved to {save_config_file}")


with st.beta_expander("Volume plot"):
    fig, ax = plt.subplots(figsize=(config["fig_width"], config["fig_height"]))
    # -- line
    for i in range(len(df_volumes_concat.columns)):
        ax.plot(
            df_volumes_concat.index,
            df_volumes_concat.iloc[:,i],
            label=df_volumes_concat.columns[i],
            marker=config["markers"][i],
            linestyle=config["line_styles"][i],
            linewidth=config["line_widths"][i],
            markersize=config["marker_sizes"][i],
            color=config["line_colors"][i],
            markerfacecolor=config["marker_facecolors"][i],
            markeredgecolor=config["marker_edgecolors"][i],
        )
    # -- title
    ax.set_title(config["title"], {"fontsize": config["title_size"],"color": config["title_color"]})
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
    if config["show_legend"]:
        plt.legend(loc=config["legend_loc"], fontsize=config["legend_size"])

    # -- show plot
    st.pyplot(plt)
    # -- save plot
    outfile = st.text_input("Save plot as:", value=os.path.join(os.path.abspath(os.getcwd()), "plot.eps"))
    if st.button("Save plot"):
        if (outfile[-4:] 
            in plt.gcf().canvas.get_supported_filetypes().keys()
            or outfile[-3:]
            in plt.gcf().canvas.get_supported_filetypes().keys()
        ):
            with st.spinner("Saving plot..."):
                plt.savefig(outfile)
            st.info(f"Plot saved to {outfile}")
        else:
            st.error(
                f"Filename should end with one of the following extensions: {list(plt.gcf().canvas.get_supported_filetypes().keys())}"
            )        