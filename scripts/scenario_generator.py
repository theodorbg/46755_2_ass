# Standard library imports
import os
import json

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Local imports
from import_data import df_wind, df_price, df_conditions
import plot_functions as pf

# Create balancing prices DataFrames to store condition-specific pricing
num_conditions = df_conditions.shape[1] #4 conditions
balancing_prices_list = [] #list to store each balancing price df

# Create balancing prices
for condition in range(num_conditions):
    df_balancing_price = df_price.copy() 
    
    for hour in range(df_conditions.shape[0]): #24 hours
        condition_value = df_conditions.iloc[hour, condition] # 0 or 1
        for day in range(df_price.shape[1]): # 20 days
            if condition_value == 0: # Excess condition
                df_balancing_price.iloc[hour, day] = df_price.iloc[hour, day] * 1.25
            else: # Deficit condition
                df_balancing_price.iloc[hour, day] = df_price.iloc[hour, day] * 0.85
    
    balancing_prices_list.append(df_balancing_price)

# Generate scenarios with hour-by-hour DataFrames
in_sample_scenarios = {}
out_of_sample_scenarios = {}
scenario_counter = 1

# Loop through each condition
for condition in range(df_conditions.shape[1]):
    for wind_day in range(df_wind.shape[1]):
        for price_day in range(df_price.shape[1]):
            # Create a DataFrame for this scenario with 24 rows (hours)
            scenario_data = pd.DataFrame(index=pd.Index(range(24), name='hour'))
            
            # Add condition column 
            scenario_data['condition'] = df_conditions.iloc[:, condition].values
            
            # Add price data
            scenario_data['price'] = df_price.iloc[:, price_day].values
            
            # Add balancing price data
            scenario_data['balancing_price'] = balancing_prices_list[condition].iloc[:, price_day].values
            
            # Add wind data
            scenario_data['wind'] = df_wind.iloc[:, wind_day].values
            
            # Store metadata as DataFrame attributes
            scenario_data.attrs = {
                'condition_id': condition,
                'wind_day': wind_day,
                'price_day': price_day
            }
            
            # Add to appropriate collection
            if scenario_counter <= 200:
                in_sample_scenarios[scenario_counter] = scenario_data
            else:
                out_of_sample_scenarios[scenario_counter] = scenario_data
                
            scenario_counter += 1

# Save scenarios to pickle files (best for preserving DataFrame structure)
# Create directory if it doesn't exist
os.makedirs('results/scenarios', exist_ok=True)

# Save in-sample scenarios
with open('results/scenarios/in_sample_scenarios.pkl', 'wb') as f:
    pickle.dump(in_sample_scenarios, f)

# Save out-of-sample scenarios
with open('results/scenarios/out_of_sample_scenarios.pkl', 'wb') as f:
    pickle.dump(out_of_sample_scenarios, f)

# Example: Access scenario 1
# print(f"Scenario 1 data:")
# print(in_sample_scenarios[1])
# print('amount of in_sample_scenarios:', len(in_sample_scenarios))
# print('amount of out_of_sample_scenarios:', len(out_of_sample_scenarios))


# Plot the first 5 scenarios (wind, price, balancing price)
for i in range(1, 6):
    pf.plot_scenario(in_sample_scenarios[i], i)

# Generate  balancing price, and day ahead price (20 plots)
pf.plot_balancing_prices_by_day(balancing_prices_list, df_price)