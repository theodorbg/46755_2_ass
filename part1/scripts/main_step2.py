print('\nImporting modules for main_step2.py')
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
# import steps_functions.step3_expost_analysis as s3 
# from steps_functions.step3_expost_analysis import perform_cross_validation, calculate_profits
from main_step1 import optimal_offers_one_price, expected_profit_one_price, scenario_profits_one_price
import data_and_scenario_generation.scenario_generator 

print('#################################')
print('\nInitializing step2.py: TWO-PRICE BALANCING ...')

# Read in_sample and out of_sample scenarios
in_sample_scenarios, out_sample_scenarios = load_scenarios.load_scenarios()

#Define constants
CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0
N_HOURS = in_sample_scenarios[0].shape[0]  # 24 hours
# %%  1.2 Offering Strategy Under a Two-Price Balancing Scheme

# Solve the two-price model
print('\nSolving the model for Two-Price offering strategy')
print('####################################################################')
(optimal_offers_two_price,
 two_price_total_expected_profit,
 two_price_scenario_profits)  = (s2.solve_two_price_offering_strategy(
     in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS))

print('\nSolved the model for Two-Price offering strategy')
print('####################################################################')

# Print results
print("\n=== TWO-PRICE BALANCING SCHEME RESULTS ===")
print(f"\nExpected profit (Two_Price): {two_price_total_expected_profit:.2e} EUR")

# Plot optimal offers
pf.plot_optimal_offers(optimal_offers_two_price, title="Optimal Day-Ahead Offers - Two-Price Scheme", 
                     filename="optimal_offers_two_price.png")

# Plot profit distribution
try:
    two_price_scenario_profits
    # If scenario profits are available, plot the cumulative distribution function
    pf.plot_cumulative_distribution_func(two_price_scenario_profits, two_price_total_expected_profit, 'Two-Price')
except NameError:
    print("Scenario profits not available for two-price scheme. Skipping plot. \nRemember to save these so we can make the plot")

# Compare with one-price results
s2.compare_one_price_two_price(expected_profit_one_price, two_price_total_expected_profit)

# Plot comparison of offering strategies
pf.compare_offers(optimal_offers_one_price, optimal_offers_two_price)

# %%
print('#################################')
# Add this section after the two-price results
print("\n=== FORECAST STRATEGY ===")
print('Analysis to see if bidding the expected wind production is better ' \
'than the gurobi optimization')

# ew = expected wind production
ew_optimal_offers, ew_expected_profit, ew_scenario_profits = (
    s2.forecast_strategy(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS))

# Compare with other strategies
s2.compare_all_strategies(expected_profit_one_price,
                           two_price_total_expected_profit,
                           ew_expected_profit)

# Plot comparison of all three strategies
pf.compare_all_strategies(optimal_offers_one_price,
                          optimal_offers_two_price,
                          ew_optimal_offers)

# Plot profit distribution
pf.plot_cumulative_distribution_func(two_price_scenario_profits, two_price_total_expected_profit, 'Two-Price')
print('\nPlotted profit distribution')

def compare_profit_distributions(one_price_profits, two_price_profits, one_price_exp, two_price_exp):
    plt.figure(figsize=(12, 6))
    
    # Plot One-Price CDF
    sorted_profits_one = np.sort(list(one_price_profits.values()))
    p_one = 1. * np.arange(len(sorted_profits_one)) / (len(sorted_profits_one) - 1)
    plt.plot(sorted_profits_one, p_one, 'b-', label='One-Price')
    
    # Plot Two-Price CDF
    sorted_profits_two = np.sort(list(two_price_profits.values()))
    p_two = 1. * np.arange(len(sorted_profits_two)) / (len(sorted_profits_two) - 1)
    plt.plot(sorted_profits_two, p_two, 'g-', label='Two-Price')
    
    # Add expected profit vertical lines
    plt.axvline(x=one_price_exp, color='b', linestyle='--', 
                label=f'One-Price Exp. Profit: {one_price_exp:.3e} EUR')
    plt.axvline(x=two_price_exp, color='g', linestyle='--', 
                label=f'Two-Price Exp. Profit: {two_price_exp:.3e} EUR')
    
    plt.xlabel('Profit (EUR)', fontsize=22)
    plt.ylabel('Cumulative Probability', fontsize=22)
    plt.title('Comparison of Profit Distributions', fontsize=22)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig('part1/results/step2/figures/profit_distribution_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_profit_distributions(one_price_profits, two_price_profits):
    # Convert to arrays for analysis
    profits_one = np.array(list(one_price_profits.values()))
    profits_two = np.array(list(two_price_profits.values()))
    
    # Calculate key statistics
    metrics = {
        'Mean': [np.mean(profits_one), np.mean(profits_two)],
        'Standard Deviation': [np.std(profits_one), np.std(profits_two)],
        'Min': [np.min(profits_one), np.min(profits_two)],
        'Max': [np.max(profits_one), np.max(profits_two)],
        '5th Percentile': [np.percentile(profits_one, 5), np.percentile(profits_two, 5)],
        '95th Percentile': [np.percentile(profits_one, 95), np.percentile(profits_two, 95)],
        'Range': [np.max(profits_one) - np.min(profits_one), 
                 np.max(profits_two) - np.min(profits_two)],
        'Coefficient of Variation': [np.std(profits_one)/np.mean(profits_one), 
                                    np.std(profits_two)/np.mean(profits_two)]
    }
    
    # Create and display results table
    df_metrics = pd.DataFrame(metrics, index=['One-Price', 'Two-Price'])
    print("\n=== Profit Distribution Analysis ===")
    print(df_metrics)
    return df_metrics


compare_profit_distributions(scenario_profits_one_price, two_price_scenario_profits, expected_profit_one_price, two_price_total_expected_profit)
analyze_profit_distributions(scenario_profits_one_price, two_price_scenario_profits)

print('\nStep 2 completed')
print('#################################')

