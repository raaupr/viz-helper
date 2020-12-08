import glob
import logging
import ntpath
import os

import fire
import numpy as np
import pandas as pd
import xlsxwriter

import sys
from nicelog.formatters import Colorful

# Setup a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Setup a handler, writing colorful output
# to the console
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(Colorful())
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


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


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def process_file(file_path, output_path, debug):
    logger.info(f'Processing "{file_path}"...')
    try:
        # -- read
        df = pd.read_excel(file_path, header=0)
        # -- process
        time = df.iloc[:, 0]
        res = {}
        for i in range(1, len(df), 6):
            df_slice = df.iloc[:, i : i + 6]
            df_slice[time.name] = time
            df_slice = df_slice.set_index(time.name)
            name = df_slice.columns[0]
            df_slice = df_slice.applymap(clean)
            df_slice["min"] = df_slice.apply(min, axis=1)
            df_slice["max"] = df_slice.apply(max, axis=1)
            df_slice["delta"] = df_slice["max"] - df_slice["min"]
            df_res = df_slice[["min", "max", "delta"]]
            res[name] = df_res
            assert(len(df) == len(df_res))
        # -- write
        file_name = path_leaf(file_path)
        file_name = os.path.splitext(file_name)[0]
        file_name = file_name + "_processed.xlsx"
        output_file = os.path.join(output_path, file_name)
        workbook = xlsxwriter.Workbook(output_file)
        worksheet = workbook.add_worksheet()
        header_format = workbook.add_format({"align": "center", "valign": "vcenter"})
        # -- time
        # header
        worksheet.merge_range(0, 0, 1, 0, df.columns[0], header_format)
        worksheet.write(1, 0, df.columns[0])
        # data
        for t in time:
            worksheet.write_column(2, 0, time)
        # -- each film group
        for i, name in enumerate(res.keys()):
            # header
            worksheet.merge_range(0, (i * 3) + 1, 0, (i * 3) + 3, name, header_format)
            worksheet.write(1, (i * 3) + 1, "Min", header_format)
            worksheet.write(1, (i * 3) + 2, "Max", header_format)
            worksheet.write(1, (i * 3) + 3, "Delta", header_format)
            # data
            df_res = res[name].replace(np.nan, "", regex=True)
            worksheet.write_column(2, (i * 3) + 1, df_res["min"])
            worksheet.write_column(2, (i * 3) + 2, df_res["max"])
            worksheet.write_column(2, (i * 3) + 3, df_res["delta"])
        workbook.close()
        logger.info(f'Success: processed file saved as "{output_file}"')
    except Exception as e:
        logger.error(f'Unable to process file "{file_path}", skipping!', exc_info=debug)


def main(input_path, output_path, debug=False):
    logger.info(f'Input path: "{input_path}"')
    logger.info(f'Output path: "{output_path}"')
    if not os.path.exists(input_path):
        logger.error(f'"{input_path}" do not exist!')
    else:
        if not os.path.exists(output_path):
            logger.info(f'"{output_path}" do not exist, creating...')
            os.makedirs(output_path)
    if os.path.isfile(input_path):
        process_file(input_path, output_path, debug)
    elif os.path.isdir(input_path):
        for input_file in glob.glob(os.path.join(output_path, "*.xlsx")):
            process_file(input_file, output_path, debug)


if __name__ == "__main__":
    fire.Fire(main)