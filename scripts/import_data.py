import os

import pandas as pd
import numpy as np

from scenario_generation import df_conditions


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
        dfs[key] = pd.read_csv(path, sep = ";",
                               parse_dates=['startTime', 'endTime'])

# defining the dataframes
df_wind_actual_gen = dfs['wind_actual_gen']
df_wind_forecast = dfs['wind_forecast']
df_price_day_ahead = dfs['price_day_ahead']

#create index for hour 0 to 24 in df_price_day_ahead
df_price_day_ahead['Hour'] = df_price_day_ahead.index
df_price_day_ahead = df_price_day_ahead.set_index('Hour')

#delete "time" column in df_price_day_ahead
df_price_day_ahead = df_price_day_ahead.drop(columns=['time'])

#reverse the column order in df_price_day_ahead
df_price_day_ahead = df_price_day_ahead.iloc[:, ::-1]

# Name each column Day x, where x is the day number
for i in range(len(df_price_day_ahead.columns)):
    df_price_day_ahead.rename(columns={df_price_day_ahead.columns[i]: f'Day {i+1}'}, inplace=True)

df_price = df_price_day_ahead.copy()
print(df_price  )


# resample df_wind_actual_gen to 1 hour frequency
# Set the 'startTime' column as the index
df_wind_actual_gen.set_index('startTime', inplace=True)

# Select only numeric columns for resampling
numeric_columns = df_wind_actual_gen.select_dtypes(include=['number'])

# Resample to 1-hour frequency and sum the values
df_resampled = numeric_columns.resample('1h').sum()

# Reset the index to work with the data
df_resampled = df_resampled.reset_index()

# Add an 'Hour' column to represent the hour of the day
df_resampled['Hour'] = df_resampled['startTime'].dt.hour

# Reshape the DataFrame to have 24 rows per column
reshaped_data = df_resampled['Wind power generation - 15 min data'].values.reshape(-1, 24).T

# Create a new DataFrame with columns for each 24-hour period
df_restructured = pd.DataFrame(reshaped_data, columns=[f'Day {i+1}' for i in range(reshaped_data.shape[1])])
df_wind = df_restructured.copy()
# Display the restructured DataFrame
print(df_wind)

