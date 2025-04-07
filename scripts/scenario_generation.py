import os
import numpy as np
import pandas as pd

import scenario_generation_functions as sgf

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# Generate power system conditions
n_scenarios = 4
deficit_probability = 0.5
power_conditions_df = sgf.generate_power_system_conditions(
    n_scenarios=n_scenarios, 
    deficit_prob=deficit_probability, 
    random_seed=42
)

# Load day ahead prices to df

day_ahead_prices_raw = pd.read_excel(
    os.path.join(DATA_DIR, 'price_dayahead.xlsx'), 
    engine='openpyxl',
    decimal=',',  # Handle comma as decimal separator
    index_col=0
)



# Select all 20 days of data (reversing order to get day 1-20)
all_days_prices = day_ahead_prices_raw.iloc[:, ::-1].values

# Create scenarios for each day
all_scenarios = []
for day in range(20):  # Process each day
    day_prices = all_days_prices[:, day]
    # Reshape to match scenario format
    day_scenarios = np.tile(day_prices, (n_scenarios, 1))
    day_df = pd.DataFrame(
        day_scenarios,
        columns=[f"Hour_{i+1}" for i in range(24)],
        index=[f"Day_{day+1}_Scenario_{i}" for i in range(n_scenarios)]
    )
    all_scenarios.append(day_df)

# Combine all scenarios
day_ahead_df = pd.concat(all_scenarios)

# Generate system conditions for all scenarios
power_conditions_df = sgf.generate_power_system_conditions(
    n_scenarios=n_scenarios,  # 4 scenarios × 20 days
    deficit_prob=deficit_probability,
    random_seed=42
)

# Transpose power_conditions_df to match day_ahead_df shape
power_conditions_transposed_df = power_conditions_df.T

#rename the transposed df index to hours from 0 to 23
power_conditions_transposed_df.index = [f"{i}" for i in range(24)]

print('power conditions df')
print(power_conditions_transposed_df)

#rename index to 'Hour'
power_conditions_transposed_df.index.name = 'Hour'
#rename columns to 'Scenario no 1', 'Scenario no 2', 'Scenario no 3', 'Scenario no 4'
power_conditions_transposed_df.columns = [f'Scenario {i+1}' for i in range(n_scenarios)]



df_conditions = power_conditions_transposed_df.copy()

print(df_conditions)
# divide power_conditions_transposed_df into 4 scenarios with 20 days each
# df_conditions_0 = power_conditions_transposed_df.iloc[0]
# df_conditions_1 = power_conditions_transposed_df.iloc[:, 20:40]
# df_conditions_2 = power_conditions_transposed_df.iloc[:, 40:60]
# df_conditions_3 = power_conditions_transposed_df.iloc[:, 60:80]


#display each scenario
# print('Scenario 1')
# print(df_conditions_0)
# print('Scenario 2')
# print(df_conditions_1)
# print('Scenario 3')
# print(df_conditions_2)
# print('Scenario 4')
# print(df_conditions_3)



# Generate balancing prices
# balancing_prices_df = sgf.generate_balancing_prices(day_ahead_df, power_conditions_df)

# Print results
# print("\nPower System Conditions (1 = deficit, 0 = excess):")
# print(power_conditions_df)
# print("\nDay-Ahead Prices (showing first few rows):")
# print(day_ahead_df.head())
# print("\nBalancing Prices (showing first few rows):")
# print(balancing_prices_df.head())

# # Print summary statistics
# print("\nSummary Statistics:")
# print(f"Total number of scenarios: {len(day_ahead_df)}")
# print(f"Number of days: 20")
# print(f"Scenarios per day: {n_scenarios}")