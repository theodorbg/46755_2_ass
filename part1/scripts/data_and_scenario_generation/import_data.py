import os
import pandas as pd
import numpy as np
from generate_conditions import generate_conditions

def load_data():
    """Load and process price and wind data, generate conditions"""
    # Fix the path by going up two levels (to project root) then to data folder
    current_file_dir = os.path.dirname(os.path.abspath(__file__))  # Get current file directory
    project_root = os.path.dirname(os.path.dirname(current_file_dir))  # Go up two levels
    DATA_DIR = os.path.join(project_root, 'data')  # Path to data folder
    
    # Check if path exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}")
        
    price_path = os.path.join(DATA_DIR, 'price_dayahead.xlsx')
    wind_path = os.path.join(DATA_DIR, 'wind_actual_gen.csv')
    
    # Print path for debugging
    print(f"Looking for price data at: {price_path}")
    print(f"Looking for wind data at: {wind_path}")
    
    # Check if files exist
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Price data file not found: {price_path}")
    if not os.path.exists(wind_path):
        raise FileNotFoundError(f"Wind data file not found: {wind_path}")
    
    # Load price data and reshape
    df_price = pd.read_excel(price_path)
    df_price.index.name = 'Hour'
    df_price = df_price.drop(columns=['time']).iloc[:, ::-1]  # Drop time, reverse columns
    df_price.columns = [f'Day {i+1}' for i in range(len(df_price.columns))]
    
    # Load wind data and reshape
    df_wind_raw = pd.read_csv(wind_path, sep=";", parse_dates=['startTime'])
    df_wind_raw.set_index('startTime', inplace=True)
    
    # Resample wind data to hourly and reshape to days
    hourly_wind = df_wind_raw['Wind power generation - 15 min data'].resample('1h').mean()
    reshaped_wind = hourly_wind.values.reshape(-1, 24).T  # Transpose to have hours as rows
    df_wind = pd.DataFrame(
        reshaped_wind, 
        columns=[f'Day {i+1}' for i in range(reshaped_wind.shape[1])]
    )
    
    # Generate conditions dataframe
    df_conditions = generate_conditions()
    
    return df_wind, df_price, df_conditions

# Generate the three required dataframes
df_wind, df_price, df_conditions = load_data()


#find max value of wind production of in_sample_scenarios
CAPACITY_WIND_FARM = 500 # MW
MAX_CAPACITY_FINGRID = 8411 # MW
#normalize the wind production based on the max wind production in the in_sample_scenarios and the max capacity (500MW)
for row in df_wind.columns:
    df_wind[row] = df_wind[row] / MAX_CAPACITY_FINGRID * CAPACITY_WIND_FARM
    
# print(f"Normalized wind production in scenario 1:")
# print(df_wind.head())
