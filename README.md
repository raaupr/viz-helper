# viz-helper
Data visualization scripts for various projects.

## Getting Started
1. Install [python](https://www.python.org/downloads/) (tested with 3.9.0)
2. Download (or [clone](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository)) this repository: <br/>
    Download: 
    - Code > Download ZIP
    - Unzip the zip file
3. Install requirements (in terminal, inside the unzipped/cloned folder):
    ```bash
    pip install -r requirements
    ```

## Run the script
Run these commands in the terminal in the unzipped/cloned folder.

### Chromosome Distance & Spindle Length vs Time: `plot_distance_length.py`
```bash
python plot_distance_length.py "path_to_distance_file" "path_to_length_file"
```
Replace `path_to_distance_file` and `path_to_length_file` to the full/relative path of your data files.

To know more options you can use to customize the plot, run:
```bash
python plot_distance_length.py --help
```
Example command with customization options:
```bash
python plot_distance_length.py "~/Documents/distance.xlsx" "~/Documents/length.xlsx" -outfile="~/Documents/plot200.jpg" -distance_min=-200 
```
will use data from `"~/Documents/distance.xlsx"` and `"~/Documents/length.xlsx"` and create the plot from distance `-200` to `0` and save the resulting plot to `"~/Documents/plot200.jpg"`.
