# Standard library imports
import os

# Third-party imports
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from import_data import df_wind, df_price, df_conditions


CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0

# objective: formulate and solve optimization problem to determine its optimal offering strategy in terms of production quantity in the day-ahead market
# ANALYSIS SPAN: 24 HOURS
# MARKETS: day-ahead + balancing markets
# no reserve / intra-day markets

#uncertainty sources (hourly basis)
# 1.  1. Wind power production,
# 2. Day-ahead market price,
# 3. The real-time power system condition (whether the system experiences a supply deficit or excess)

# uncertainties assumed uncorrelated

# # Assuming the following DataFrames:
# conditions_0_df, conditions_1_df, conditions_2_df, conditions_3_df
# df_wind_actual_gen (wind power production)
# df_price_day_ahead (day-ahead prices)

# Initialize scenario counter
scenario_counter = 1

in_sample_scenarios = 

# Loop through each condition (column) in df_conditions
for condition in range(df_conditions.shape[1]):
    # Loop through each day (column) in wind power production
    for wind_day in range(df_wind.shape[1]):
        # Loop through each day (column) in day-ahead prices
        for price_day in range(df_price.shape[1]):
            # Print scenario details
            print(f"Scenario {scenario_counter}")
            print(f"Condition {condition}")
            print(f"Wind day {wind_day}")
            print(f"Day-ahead price day {price_day}")
            
            # Increment scenario counter
            scenario_counter += 1