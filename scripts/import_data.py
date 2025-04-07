import os

import pandas as pd

# Read files from \data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
files = {
    'price_day_ahead': os.path.join(DATA_DIR, 'price_dayahead.xlsx'),
    'wind_actual_gen': os.path.join(DATA_DIR, 'wind_actual_gen.csv'), 
    'wind_forecast': os.path.join(DATA_DIR, 'wind_forecast.csv')
}

# Check if files exist
for key, filepath in files.items():
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    

# Read dataframes with appropriate method based on file extension
dfs = {}
for key, path in files.items():
    if path.endswith('.xlsx'):
        dfs[key] = pd.read_excel(path)
    else:
        dfs[key] = pd.read_csv(path)

# defining the dataframes
df_wind_actual_gen = dfs['wind_actual_gen']
df_wind_forecast = dfs['wind_forecast']
df_price_day_ahead = dfs['price_day_ahead']