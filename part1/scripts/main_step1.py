print('\nImporting modules for main_step1.py')
# Standard library imports
import os
import pickle

# Third-party imports
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
import data_and_scenario_generation.load_scenarios as load_scenarios
import steps_functions.step1_one_price as s1
import steps_functions.step2_two_price as s2
import steps_functions.plot_functions as pf
import steps_functions.step3_expost_analysis as s3 
from steps_functions.step3_expost_analysis import perform_cross_validation, calculate_profits
import data_and_scenario_generation.scenario_generator 

# %% 1.1 Offering Strategy Under a One-Price Balancing Scheme
print('#################################')
print('\nInitializing main_step1.py: ONE-PRICE BALANCING...')
print('')

# Read in_sample and out of_sample scenarios
in_sample_scenarios, out_sample_scenarios = load_scenarios.load_scenarios()


CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0
N_HOURS = in_sample_scenarios[0].shape[0]  # 24 hours
# objective: formulate and solve optimization problem to determine its optimal offering strategy in terms of production quantity in the day-ahead market
# ANALYSIS SPAN: 24 HOURS
# MARKETS: day-ahead + balancing markets
# no reserve / intra-day markets

#uncertainty sources (hourly basis)
# 1.  1. Wind power production,
# 2. Day-ahead market price,
# 3. The real-time power system condition (whether the system experiences a supply deficit or excess)

# uncertainties assumed uncorrelated

# %%  1.1 Offering Strategy Under a One-Price Balancing Scheme

# Solve the model
print('\nSolving the model for One-Price offering strategy')
print('####################################################################')
optimal_offers_one_price, expected_profit_one_price, scenario_profits_one_price = s1.solve_one_price_offering_strategy(in_sample_scenarios,
                                                                                         CAPACITY_WIND_FARM,
                                                                                         N_HOURS)
print('\nSolved the model for One-Price offering strategy')
print('####################################################################')

# Print results
print("\n=== ONE-PRICE BALANCING SCHEME RESULTS ===")

print(f"\nExpected profit (One-Price): {expected_profit_one_price:.2e} EUR")
# print("Optimal day-ahead offers (MW):")
# for h in range(24):
#     print(f"Hour {h}: {optimal_offers_one_price[h]:.2e} MW")

# Plot optimal offers
pf.plot_optimal_offers(optimal_offers_one_price)
print('\nPlotted optimal offers')
 
# Plot profit distribution
pf.plot_cumulative_distribution_func(scenario_profits_one_price, expected_profit_one_price, 'One-Price')
print('\nPlotted profit distribution')

# Analyze if we see an all-or-nothing bidding strategy
threshold = 1e-6  # MW, to account for potential numerical precision
all_or_nothing = all(offer <= threshold or abs(offer - CAPACITY_WIND_FARM) <= threshold for offer in optimal_offers_one_price)
print(f"All-or-nothing bidding strategy: {'Yes' if all_or_nothing else 'No'}")

print('\nStep 1 completed')
print('#################################')
