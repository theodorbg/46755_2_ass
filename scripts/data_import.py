import pandas as pd
import os
import matplotlib.pyplot as plt

# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd


# Get the project root directory (2 levels up from script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(project_root, 'data')

# File configuration with absolute paths
files = {
    'price_day_ahead': os.path.join(data_dir, 'Price_dayahead.xlsx'),
    'wind_actual_gen': os.path.join(data_dir, 'wind_actual_gen.csv'), 
    'wind_forecast': os.path.join(data_dir, 'wind_forecast.csv')
}

# Search for data files and build paths dictionary
paths = {}
for key, filepath in files.items():
    if os.path.exists(filepath):
        paths[key] = filepath
    else:
        print(f"Warning: Could not find {filepath}")

# Read dataframes with appropriate method based on file extension
dfs = {}
for key, path in paths.items():
    if path.endswith('.xlsx'):
        dfs[key] = pd.read_excel(path)
    else:
        dfs[key] = pd.read_csv(path)