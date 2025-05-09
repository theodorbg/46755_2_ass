# Standard library imports
import os
import json

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from import_data import df_wind, df_price, df_conditions
import plot_functions as pf

# Create balancing prices DataFrames to store condition-specific pricing
num_conditions = df_conditions.shape[1] #4 conditions
balancing_prices_list = [] #list to store each balancing price df

# %% Create balancing prices
# Loop through each condition
for condition in range(num_conditions):
    # copy df_prices since balancing prices just have a factor applied to each value in df_prices
    df_balancing_price = df_price.copy() 
    
    # Apply balancing price rules for this condition
    # loop through each hour and day in df_price (24x20)
    for hour in range(df_conditions.shape[0]): #24 hours
        condition_value = df_conditions.iloc[hour, condition] # 0 or 1
        # Loop through each day in df_price (20 days)
        for day in range(df_price.shape[1]): # 20 days
            # Apply the balancing price rules based on the condition value: 0 or 1            
            if condition_value == 0: # If condition_value is 0, increase the price by 25%
                df_balancing_price.iloc[hour, day] = df_price.iloc[hour, day] * 1.25
            else: # If condition_value is 1, decrease the price by 15%
                df_balancing_price.iloc[hour, day] = df_price.iloc[hour, day] * 0.85
    
    # Store the processed DataFrame for this condition
    balancing_prices_list.append(df_balancing_price)

# %%
# Generate scenarios
scenario_counter = 1
in_sample_scenarios = []
out_of_sample_scenarios = []

# Loop through each condition (4 conditions for 24 hours)
for condition in range(df_conditions.shape[1]):
    # Loop through each wind day
    for wind_day in range(df_wind.shape[1]):
        # Loop through each price day (20 days)
        for price_day in range(df_price.shape[1]):
            # Use the pre-calculated balancing prices for this condition
            df_balancing_prices = balancing_prices_list[condition]
            
            # Store the complete scenario
            scenario = {
                "Scenario": scenario_counter,
                #"Condition": condition,
                "Wind Day": wind_day,
                "Price Day": price_day,
                "Balancing Prices": df_balancing_prices.iloc[:, price_day].values.tolist()
            }
            
            # Add to appropriate collection
            if scenario_counter <= 200:
                in_sample_scenarios.append(scenario)
            else:
                out_of_sample_scenarios.append(scenario)
                
            scenario_counter += 1

# Convert the lists of Scenarios to DataFrames
df_in_sample_scenarios = pd.DataFrame(in_sample_scenarios)
df_out_of_sample_scenarios = pd.DataFrame(out_of_sample_scenarios)

# Save the DataFrames to CSV files
df_in_sample_scenarios.to_csv('in_sample_scenarios.csv', index=False)
df_out_of_sample_scenarios.to_csv('out_of_sample_scenarios.csv', index=False)

print(df_in_sample_scenarios['Balancing Prices'])
print(df_in_sample_scenarios.columns)
      

# Visualize every price and balancing price for each day 
days = len(df_price.columns)  # Number of days in df_price (20 days)
for day in range(days):
    pf.plot_balancing_prices(balancing_prices_list, df_price, day)


# %%
# Generate scenarios as dictionaries
in_sample_scenarios = {}
out_of_sample_scenarios = {}
scenario_counter = 1

# Loop through each condition
for condition in range(df_conditions.shape[1]):
    for wind_day in range(df_wind.shape[1]):
        for price_day in range(df_price.shape[1]):
            # Use the pre-calculated balancing prices for this condition
            df_balancing_prices = balancing_prices_list[condition]
            
            # Create scenario dictionary
            scenario = {
                "wind_data": df_wind.iloc[:, wind_day].values.tolist(),
                "price_data": df_price.iloc[:, price_day].values.tolist(),
                "balancing_price_data": df_balancing_prices.iloc[:, price_day].values.tolist(),
                "condition": condition,
                "wind_day": wind_day,
                "price_day": price_day
            }
            
            # Add to appropriate collection
            if scenario_counter <= 200:
                in_sample_scenarios[scenario_counter] = scenario
            else:
                out_of_sample_scenarios[scenario_counter] = scenario
                
            scenario_counter += 1


# Save to JSON files
with open('in_sample_scenarios.json', 'w') as f:
    json.dump(in_sample_scenarios, f)
    
with open('out_of_sample_scenarios.json', 'w') as f:
    json.dump(out_of_sample_scenarios, f)

# Get wind data for scenario 5
wind_data_scenario_5 = in_sample_scenarios[5]["wind_data"]

# get all prices for each scenario in in_sample_scenarios

prices = [in_sample_scenarios[i]["price_data"] for i in in_sample_scenarios.keys()]
print('first price scenario')
print(prices[0])


def plot_balancing_prices(scenarios_dict, df_price, price_day):
    """
    Plot balancing prices for all conditions on a specific price day.
    
    Args:
        scenarios_dict: Dictionary of scenarios with the new structure
        df_price: Original price DataFrame
        price_day: The price day to visualize
    """
    plt.figure(figsize=(14, 8))
    
    # Get unique conditions
    conditions = sorted(list(set(scenario["condition"] for scenario in scenarios_dict.values())))
    colors = ['blue', 'orange', 'green', 'red']
    
    # For each condition, find a scenario with the specified price day
    for condition in conditions:
        # Find the first scenario with this condition and price day
        matching_scenarios = [
            s for s_id, s in scenarios_dict.items() 
            if s["condition"] == condition and s["price_day"] == price_day
        ]
        
        if matching_scenarios:
            scenario = matching_scenarios[0]
            balancing_prices = scenario["balancing_price_data"]
            
            # Plot with thicker lines and markers for better visibility
            plt.plot(balancing_prices, label=f'Condition {condition}', 
                     color=colors[condition % len(colors)], 
                     linewidth=2.5, marker='o', markersize=5, alpha=0.7)
    
    # Plot the original price data for this day
    plt.plot(df_price.iloc[:, price_day], label='Day-Ahead Price', 
             color='black', linestyle='--', linewidth=2)
    
    # Add details about conditions
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.annotate('Condition 0: Excess (Price +25%)', xy=(0.02, 0.97), xycoords='axes fraction', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.annotate('Condition 1: Deficit (Price -15%)', xy=(0.02, 0.93), xycoords='axes fraction', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f'Balancing Prices by Condition - Day {price_day+1}', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Price (EUR/MWh)', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))  # Show every 2 hours for clarity
    
    # Create directory if it doesn't exist
    os.makedirs('results/figures/balancing_prices', exist_ok=True)
    plt.savefig(f'results/figures/balancing_prices/balancing_prices_day_{price_day+1}.png', 
                dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
# Example usage
days = len(df_price.columns)  # Number of days in df_price (20 days)
for day in range(days):
    plot_balancing_prices(in_sample_scenarios, df_price, day)
# condition_2_scenarios = {k: v for k, v in in_sample_scenarios.items() if v["condition"] == 2}

# print("All scenarios with condition 2:")
# print(condition_2_scenarios)

# # Visualize every price and balancing price for each day 
# days = len(df_price.columns)  # Number of days in df_price (20 days)
# balancing_prices_list = in_sample_scenarios['balancing_price_data']
# for day in range(days):
#     pf.plot_balancing_prices(balancing_prices_list, df_price, day)