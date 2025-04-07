# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from import_data import df_wind, df_price, df_conditions


scenario_counter = 1

in_sample_scenarios = []
out_of_sample_scenarios = []
df_balancing_prices_0 = df_price.copy()
df_balancing_prices_1 = df_price.copy()
df_balancing_prices_2 = df_price.copy()
df_balancing_prices_3 = df_price.copy()

# Create a list of balancing prices DataFrames for each condition
df_balancing_prices_list = [
    df_balancing_prices_0,
    df_balancing_prices_1,
    df_balancing_prices_2,
    df_balancing_prices_3
]

# Loop through each condition (column) in df_conditions
for condition in range(df_conditions.shape[1]):
    # Get the corresponding balancing prices DataFrame for the current condition
    df_balancing_prices = df_balancing_prices_list[condition]

    # Loop through each day (column) in wind power production
    for wind_day in range(df_wind.shape[1]):
        # Loop through each day (column) in day-ahead prices
        for price_day in range(df_price.shape[1]):
            # Access the values in each row for the current condition
            for hour in range(df_conditions.shape[0]):  # Loop through rows (hours)
                condition_value = df_conditions.iloc[hour, condition]
                wind_value = df_wind.iloc[hour, wind_day]
                price_value = df_price.iloc[hour, price_day]

                # Print row-level details (optional)
                print(f"Hour {hour}: Condition={condition_value}, Wind={wind_value}, Price={price_value}")

                # Generate balancing prices based on the condition
                if condition_value == 0:
                    # Modify the specific cell in the current df_balancing_prices for the current price_day and hour
                    df_balancing_prices.iloc[hour, price_day] = df_price.iloc[hour, price_day] * 1.25  # if system is in deficit
                else:
                    # Modify the specific cell in the current df_balancing_prices for the current price_day and hour
                    df_balancing_prices.iloc[hour, price_day] = df_price.iloc[hour, price_day] * 0.85  # if system is in excess

            # Store the scenario
            if scenario_counter <= 200:
                in_sample_scenarios.append({
                    "Scenario": scenario_counter,
                    "Condition": condition,
                    "Wind Day": wind_day,
                    "Price Day": price_day,
                    "Balancing Prices": df_balancing_prices.values.tolist(),
                    "Hourly Values": {
                        "Condition": df_conditions.iloc[:, condition].tolist(),
                        "Wind": df_wind.iloc[:, wind_day].tolist(),
                        "Price": df_price.iloc[:, price_day].tolist()
                    }
                })
            else:
                out_of_sample_scenarios.append({
                    "Scenario": scenario_counter,
                    "Condition": condition,
                    "Wind Day": wind_day,
                    "Price Day": price_day,
                    "Balancing Prices": df_balancing_prices.values.tolist(),
                    "Hourly Values": {
                        "Condition": df_conditions.iloc[:, condition].tolist(),
                        "Wind": df_wind.iloc[:, wind_day].tolist(),
                        "Price": df_price.iloc[:, price_day].tolist()
                    }
                })

            # Increment scenario counter
            scenario_counter += 1


#Convert the lists of Scenarios to DataFrames
df_in_sample_scenarios = pd.DataFrame(in_sample_scenarios)
df_out_of_sample_scenarios = pd.DataFrame(out_of_sample_scenarios)

#Save the DataFrames to CSV files
df_in_sample_scenarios.to_csv('in_sample_scenarios.csv', index=False)
df_out_of_sample_scenarios.to_csv('out_of_sample_scenarios.csv', index=False)

# print("In-sample scenarios:")
# print(df_in_sample_scenarios)

print('Balancing Prices ')
print(df_balancing_prices_0)
print(df_balancing_prices_1)
print(df_balancing_prices_2)
print(df_balancing_prices_3)

# %%
# Visualize day 1 for each df_balancing price df
plt.figure(figsize=(12, 6))
plt.plot(df_balancing_prices_0.iloc[:, 0], label='Condition 0', color='blue')
plt.plot(df_balancing_prices_1.iloc[:, 0], label='Condition 1', color='orange')
plt.plot(df_balancing_prices_2.iloc[:, 0], label='Condition 2', color='green')
plt.plot(df_balancing_prices_3.iloc[:, 0], label='Condition 3', color='red')
plt.plot(df_price.iloc[:, 0], label='Day-Ahead Price', color='grey', linestyle='--')
plt.title('Balancing Prices for Day 1')
plt.xlabel('Hour')
plt.ylabel('Price (EUR/MWh)')
plt.legend()
plt.grid()
plt.savefig('balancing_prices_day_1.png', dpi=300, bbox_inches='tight')
plt.show()