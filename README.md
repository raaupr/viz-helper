[![DOI](https://zenodo.org/badge/316316473.svg)](https://zenodo.org/badge/latestdoi/316316473)
# viz-helper
Helper tools for visualizations of chromosomes. 

As published in: 

>Kinetochore Components Function in C. elegans Oocytes Revealed by 4D Tracking of Holocentric Chromosomes

Tools used in the paper:

- `viz_threshold`
- `viz_twoy_threshold`

## Getting Started
1. Install [python](https://www.python.org/downloads/) (tested with 3.8.5)
2. Download (or [clone](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository)) this repository: <br/>
    Download: 
    - Code > Download ZIP
    - Unzip the zip file
3. Install requirements (in terminal, inside the unzipped/cloned folder):
    ```bash
    pip install -r requirements.txt
    ```

## Run    
Run one of these commands in the terminal in the unzipped/cloned folder.
```bash
streamlit run viz.py
```
```bash
streamlit run viz_threshold.py
```
```bash
streamlit run viz_twoy.py
```
```bash
streamlit run viz_twoy_threshold.py
```
```bash
streamlit run viz_polyhedron.py
```
```bash
streamlit run viz_polyhedra.py
```
```bash
python maxmin.py INPUT_PATH OUTPUT_PATH
```
