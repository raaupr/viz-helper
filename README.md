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

### Distance + Length vs Time: `plot_distance_length.py`
Example:
```bash
python plot_distance_length.py "~/Documents/distance.xlsx" "~/Documents/length.xlsx"
```
For more options, see:
```bash
python plot_distance_length.py --help
```
For example:
```bash
python plot_distance_length.py "~/Documents/distance.xlsx" "~/Documents/length.xlsx" -outfile="~/Documents/plot200.jpg" -distance_min=-200 
```
will use data from `"~/Documents/distance.xlsx"` and `"~/Documents/length.xlsx"` and create the plot from distance `-200` to `0` and save the resulting plot to `"~/Documents/plot200.jpg"`.
