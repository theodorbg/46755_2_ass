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
import steps.step1_one_price as s1
import steps.step2_two_price as s2
import plot_functions as pf

# Read in_sample and out of_sample scenarios
in_sample_scenarios, out_sample_scenarios = load_scenarios.load_scenarios()


# Example: Access scenario 1
# print(f"Scenario 1 data:")
# print(in_sample_scenarios[1])
# print('amount of in_sample_scenarios:', len(in_sample_scenarios))
# print('amount of out_of_sample_scenarios:', len(out_sample_scenarios))


CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0
N_HOURS = in_sample_scenarios[1].shape[0]  # 24 hours


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
# formulate and solve the stochastic offering strategy problem 
# for a one-price balancing scheme using in-sample scenarios.

# Determine the optimal hourly production quantity offers of the 
# wind farm in the day-ahead market and calculate the expected profit

# Additionally, illustrate the cumulative distribution of profit 
# across the in-sample scenarios. 

# Do we observe the wind farm bidding either 0 or full capacity 
# (an all-or-nothing strategy)? If so, why?

# From slides:
# problem determines:
# how much quantity to offer
# at what price? (0 ??= offer price)
# objective: maximize expected profit

# Assumptions:
# The wind farm is a price taker in the day-ahead market
# Each scenario is independent and equally likely (1/200) for in-sample scenarios

# Solve the model
optimal_offers, expected_profit, scenario_profits = s1.solve_one_price_offering_strategy(in_sample_scenarios,
                                                                                         CAPACITY_WIND_FARM,
                                                                                         N_HOURS)

# Print results
print(f"Expected profit: {expected_profit:.2f} EUR")
# print("Optimal day-ahead offers (MW):")
# for h in range(24):
#     print(f"Hour {h}: {optimal_offers[h]:.2f} MW")

# Plot optimal offers
pf.plot_optimal_offers(optimal_offers)

# Plot profit distribution
pf.plot_cumulative_distribution_func(scenario_profits, expected_profit, 'One-Price')

# Analyze if we see an all-or-nothing bidding strategy
threshold = 1e-6  # MW, to account for potential numerical precision
all_or_nothing = all(offer <= threshold or abs(offer - CAPACITY_WIND_FARM) <= threshold for offer in optimal_offers)
print(f"All-or-nothing bidding strategy: {'Yes' if all_or_nothing else 'No'}")


# %%  1.2 Offering Strategy Under a Two-Price Balancing Scheme
#Repeat Step 1.1, but now consider a two-price balancing scheme. 
# Analyze any significant differences between the results of Step 1.1 and 
# Step 1.2, particularly in terms of the offering strategy and profit distribution

# from slides:
# yt link: https://www.youtube.com/watch?v=9dEe5JdqPp4&ab_channel=Renewablesinelectricitymarkets

# Solve the two-price model
tp_optimal_offers, tp_expected_profit, tp_scenario_profits = s2.solve_two_price_offering_strategy(
    in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS
)

# Print results
print("\n=== TWO-PRICE BALANCING SCHEME RESULTS ===")
print(f"Expected profit: {tp_expected_profit:.2f} EUR")
print("Optimal day-ahead offers (MW):")
for h in range(24):
    print(f"Hour {h}: {tp_optimal_offers[h]:.2f} MW")

# Plot optimal offers
pf.plot_optimal_offers(tp_optimal_offers, title="Optimal Day-Ahead Offers - Two-Price Scheme", 
                     filename="two_price_optimal_offers.png")

# Plot profit distribution
pf.plot_cumulative_distribution_func(tp_scenario_profits, tp_expected_profit, 
                                   'Two-Price')

# Analyze if we see an all-or-nothing bidding strategy
tp_all_or_nothing = all(offer <= threshold or abs(offer - CAPACITY_WIND_FARM) <= threshold 
                      for offer in tp_optimal_offers)
print(f"All-or-nothing bidding strategy: {'Yes' if tp_all_or_nothing else 'No'}")

# Compare with one-price results
print("\n=== COMPARISON: ONE-PRICE vs TWO-PRICE ===")
print(f"One-Price Expected Profit: {expected_profit:.2f} EUR")
print(f"Two-Price Expected Profit: {tp_expected_profit:.2f} EUR")
print(f"Difference: {tp_expected_profit - expected_profit:.2f} EUR")

# Plot comparison of offering strategies
plt.figure(figsize=(12, 6))
plt.bar(np.arange(24) - 0.2, optimal_offers, width=0.4, label='One-Price', color='blue', alpha=0.7)
plt.bar(np.arange(24) + 0.2, tp_optimal_offers, width=0.4, label='Two-Price', color='orange', alpha=0.7)
plt.xlabel('Hour')
plt.ylabel('Offer Quantity (MW)')
plt.title('Comparison of Optimal Day-Ahead Offers')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/comparison_optimal_offers.png', dpi=300, bbox_inches='tight')
plt.show()


# %% DEBUGGING 1.2
# After solving all hours
price_diff_by_hour = []
for hour in range(N_HOURS):
    DA_prices = [in_sample_scenarios[s].loc[hour, 'price'] for s in in_sample_scenarios]
    BAL_prices_excess = [in_sample_scenarios[s].loc[hour, 'balancing_price'] 
                         for s in in_sample_scenarios if in_sample_scenarios[s].loc[hour, 'condition'] == 0]
    BAL_prices_deficit = [in_sample_scenarios[s].loc[hour, 'balancing_price'] 
                          for s in in_sample_scenarios if in_sample_scenarios[s].loc[hour, 'condition'] == 1]
    
    avg_DA = sum(DA_prices) / len(DA_prices)
    avg_BAL_excess = sum(BAL_prices_excess) / len(BAL_prices_excess) if BAL_prices_excess else 0
    avg_BAL_deficit = sum(BAL_prices_deficit) / len(BAL_prices_deficit) if BAL_prices_deficit else 0
    
    price_diff_by_hour.append({
        'hour': hour,
        'avg_DA': avg_DA,
        'avg_BAL_excess': avg_BAL_excess,
        'avg_BAL_deficit': avg_BAL_deficit,
        'surplus_penalty': (avg_DA - avg_BAL_excess) / avg_DA if avg_DA else 0,
        'deficit_penalty': (avg_BAL_deficit - avg_DA) / avg_DA if avg_DA else 0,
        'optimal_bid': optimal_offers[hour]
    })

# Print analysis for hours with 0 or 500 MW bids
print("\nPrice difference analysis for extreme bidding hours:")
for data in price_diff_by_hour:
    if data['optimal_bid'] < 1 or data['optimal_bid'] > 499:
        print(f"Hour {data['hour']}: Bid = {data['optimal_bid']:.0f} MW, " +
              f"DA = {data['avg_DA']:.2f}, " +
              f"BAL(excess) = {data['avg_BAL_excess']:.2f}, " +
              f"BAL(deficit) = {data['avg_BAL_deficit']:.2f}, " +
              f"Surplus penalty = {data['surplus_penalty']*100:.0f}%, " +
              f"Deficit penalty = {data['deficit_penalty']*100:.0f}%")
        


# Add detailed debugging for problematic hours
if hour in [11, 12, 13]:
    print(f"\n--- Detailed analysis for Hour {hour} ---")
    # Check data ranges
    wind_values = [in_sample_scenarios[s].loc[hour, 'wind'] for s in in_sample_scenarios]
    price_values = [in_sample_scenarios[s].loc[hour, 'price'] for s in in_sample_scenarios]
    
    print(f"Wind min/avg/max: {min(wind_values):.1f}/{sum(wind_values)/len(wind_values):.1f}/{max(wind_values):.1f}")
    print(f"DA Price min/avg/max: {min(price_values):.2f}/{sum(price_values)/len(price_values):.2f}/{max(price_values):.2f}")
    
    # Check for extreme or invalid values
    extreme_wind = [s for s in in_sample_scenarios if abs(in_sample_scenarios[s].loc[hour, 'wind']) > 1000]
    negative_wind = [s for s in in_sample_scenarios if in_sample_scenarios[s].loc[hour, 'wind'] < 0]
    
    if extreme_wind:
        print(f"Found {len(extreme_wind)} scenarios with extreme wind values")
    if negative_wind:
        print(f"Found {len(negative_wind)} scenarios with negative wind values")
# %%  Ex-post Analysis
#  Following Lecture 8, conduct ex-post cross-validation analyses to
#  evaluate the quality of the offering decisions made in both Steps 1.1 and 1.2. 
# With 200 in-sample and 1,400 out-of-sample scenarios, 
# perform an 8-fold cross-validation analysis.
# 
#  For each run (with the given 200 in-sample and 1,400 out-of-sample scenarios), 
# calculate the expected profits for both the in-sample and out-of-sample analyses. 
# 
# After completing all 8 runs, calculate the average expected profits for both 
# the in-sample and out-of-sample analyses. 
# 
# Considering the results from all 8 runs, compare the average expected profits from
#  the in-sample analyses to those from the out-of-sample analyses. 
# 
# Based on this comparison, can we interpret how satisfactory the offering 
# decisions are? 
# 
# While keeping the total number of scenarios at 1,600, discuss whether 
# altering the number of in-sample scenarios from 200 would improve the quality 
# of the offering decisions, and if so, to what extent

# %% Risk-Averse Offering Strategy
# Following Lecture 9, formulate and solve the risk averse offering strategy 
# problem for the wind farm under both one- and two-price balancing schemes (α = 0.90). 
# 
# Gradually increase the value of β from zero and plot a two-dimensional
#  figure showing expected profit versus Conditional Value at Risk (CVaR). 
# 
# Explain how the offering strategy and profit volatility evolve as β increases. 
# 
# Additionally, discuss how the profit distribution across scenarios changes when 
# risk considerations are incorporated. 
# 
# Lastly, analyze whether changing the set and number of in-sample scenarios leads 
# to significant changes in the risk-averse offering decisions. 

# This task does not require any ex-post out-of-sample or cross-validation analyses


