import yaml


LINE_STYLES = ["solid", "dotted", "dashed", "dashdot"]
LEGEND_LOCATIONS = [
    'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
]

def write_marker_info(container):
    container.markdown(
        "<div style=' font-size: small;'>See <a href='https://matplotlib.org/api/markers_api.html'>here</a> for possible marker types</div>",
        unsafe_allow_html=True,
    )

def select_config(container, id):
    fin_config = container.file_uploader(
        "Select a new configuration file (optional)",
        type="yml",
        key=id
    )
    if fin_config:
        config = yaml.load(fin_config, Loader=yaml.Loader)
    else:
        config = None
    return config

