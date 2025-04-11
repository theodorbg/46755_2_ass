import os
import pandas as pd
from generate_conditions import generate_conditions

def load_data():
    """Load and process price and wind data, generate conditions"""
    # Set data directory and paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    price_path = os.path.join(DATA_DIR, 'price_dayahead.xlsx')
    wind_path = os.path.join(DATA_DIR, 'wind_actual_gen.csv')
    
    # Load price data and reshape
    df_price = pd.read_excel(price_path)
    df_price.index.name = 'Hour'
    df_price = df_price.drop(columns=['time']).iloc[:, ::-1]  # Drop time, reverse columns
    df_price.columns = [f'Day {i+1}' for i in range(len(df_price.columns))]
    
    # Load wind data and reshape
    df_wind_raw = pd.read_csv(wind_path, sep=";", parse_dates=['startTime'])
    df_wind_raw.set_index('startTime', inplace=True)
    
    # Resample wind data to hourly and reshape to days
    hourly_wind = df_wind_raw['Wind power generation - 15 min data'].resample('1h').sum()
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

# # For debugging only - comment out in production
# if __name__ == "__main__":
#     print("Wind data sample:")
#     print(df_wind.head())
#     print("\nPrice data sample:")
#     print(df_price.head())
#     print("\nConditions data:")
#     print(df_conditions)