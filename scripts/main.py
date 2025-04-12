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
import steps.step1_solve_one_price_offering_strategy as s1
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
print("Optimal day-ahead offers (MW):")
for h in range(24):
    print(f"Hour {h}: {optimal_offers[h]:.2f} MW")

# Plot optimal offers
pf.plot_optimal_offers(optimal_offers)

# Plot profit distribution
pf.plot_cumulative_distribution_func(scenario_profits, expected_profit)

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


