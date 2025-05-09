print('#################################')
print('\nInitializing step2.py...')
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
from main_step1 import optimal_offers_one_price, expected_profit_one_price, scenario_profits_one_price

# Read in_sample and out of_sample scenarios
in_sample_scenarios, out_sample_scenarios = load_scenarios.load_scenarios()

CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0
N_HOURS = in_sample_scenarios[0].shape[0]  # 24 hours
# %%  1.2 Offering Strategy Under a Two-Price Balancing Scheme

print('#################################')

print("\n=== TWO-PRICE BALANCING ===")
# Solve the two-price model

print('\nSolving two-price model...')
optimal_offers_two_price, two_price_total_expected_profit, two_price_scenario_profits  = s2.solve_two_price_offering_strategy(
    in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS
)

# optimal_offers_two_price, two_price_total_expected_profit, two_price_scenario_profits = s2.solve_two_price_offering_strategy_hourly(
#     in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS
# )


# Print results
print("\n=== TWO-PRICE BALANCING SCHEME RESULTS ===")
print(f"Expected profit (Two_Price): {two_price_total_expected_profit:.2e} EUR")
# print("Optimal day-ahead offers (MW):")
# for h in range(len(optimal_offers_two_price)):
#     print(f"Hour {h}: {optimal_offers_two_price[h]:.2e} MW")
# Plot optimal offers
pf.plot_optimal_offers(optimal_offers_two_price, title="Optimal Day-Ahead Offers - Two-Price Scheme", 
                     filename="optimal_offers_two_price.png")

# Plot profit distribution
# if two_price_scenario_profits exists:
    # If scenario profits are available, plot the cumulative distribution function

try:
    two_price_scenario_profits
    # If scenario profits are available, plot the cumulative distribution function
    pf.plot_cumulative_distribution_func(two_price_scenario_profits, two_price_total_expected_profit, 'Two-Price')
except NameError:
    print("Scenario profits not available for two-price scheme. Skipping plot. \nRemember to save these so we can make the plot")

# Analyze if we see an all-or-nothing bidding strategy
# tp_all_or_nothing = all(offer <= threshold or abs(offer - CAPACITY_WIND_FARM) <= threshold 
#                       for offer in optimal_offers_two_price)
# print(f"All-or-nothing bidding strategy: {'Yes' if tp_all_or_nothing else 'No'}")

# Compare with one-price results
print("\n=== COMPARISON: ONE-PRICE vs TWO-PRICE ===")
print(f"One-Price Expected Profit: {expected_profit_one_price:.2e} EUR")
print(f"Two-Price Expected Profit: {two_price_total_expected_profit:.2e} EUR")
print(f"Difference: {two_price_total_expected_profit - expected_profit_one_price:.2e} EUR")

# Plot comparison of offering strategies
pf.compare_offers(optimal_offers_one_price, optimal_offers_two_price)

# %%
print('#################################')
# Add this section after the two-price results
print("\n=== FORECAST STRATEGY ===")
ew_optimal_offers, ew_expected_profit, ew_scenario_profits = s2.forecast_strategy(
    in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS
)

# print(f"Expected profit: {ew_expected_profit:.2e} EUR")
# print("Optimal day-ahead offers (MW):")
# for h in range(24):
#     print(f"Hour {h}: {ew_optimal_offers[h]:.2e} MW")

# Compare with other strategies
print("\n=== COMPARISON: ALL STRATEGIES ===")
print(f"One-Price Expected Profit: {expected_profit_one_price:.2e} EUR")
print(f"Two-Price Expected Profit: {two_price_total_expected_profit:.2e} EUR")
print(f"Expected Wind Profit: {ew_expected_profit:.2e} EUR")
print(f"Expected Wind vs One-Price: {ew_expected_profit - expected_profit_one_price:.2e} EUR")
print(f"Expected Wind vs Two-Price: {ew_expected_profit - two_price_total_expected_profit:.2e} EUR")

# Plot comparison of all three strategies
pf.compare_all_strategies(optimal_offers_one_price, optimal_offers_two_price, ew_optimal_offers)

print('#################################')

