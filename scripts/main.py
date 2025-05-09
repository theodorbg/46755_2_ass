print('#################################')
print('\nInitializing main.py...')
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
import steps.step3_expost_analysis as s3 
from steps.step3_expost_analysis import perform_cross_validation, calculate_profits

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
optimal_offers_one_price, expected_profit, scenario_profits = s1.solve_one_price_offering_strategy(in_sample_scenarios,
                                                                                         CAPACITY_WIND_FARM,
                                                                                         N_HOURS)

# Print results
print(f"\nExpected profit (One-Price): {expected_profit:.2e} EUR")
# print("Optimal day-ahead offers (MW):")
# for h in range(24):
#     print(f"Hour {h}: {optimal_offers_one_price[h]:.2e} MW")

# Plot optimal offers
pf.plot_optimal_offers(optimal_offers_one_price)
print('\nPlotted optimal offers')
 
# Plot profit distribution
pf.plot_cumulative_distribution_func(scenario_profits, expected_profit, 'One-Price')
print('\nPlotted profit distribution')

# Analyze if we see an all-or-nothing bidding strategy
threshold = 1e-6  # MW, to account for potential numerical precision
all_or_nothing = all(offer <= threshold or abs(offer - CAPACITY_WIND_FARM) <= threshold for offer in optimal_offers_one_price)
print(f"All-or-nothing bidding strategy: {'Yes' if all_or_nothing else 'No'}")

# %%  1.2 Offering Strategy Under a Two-Price Balancing Scheme

print('#################################')

print("\n=== TWO-PRICE BALANCING ===")
# Solve the two-price model

print('\nSolving two-price model...')
optimal_offers_two_price, two_price_total_expected_profit  = s2.solve_two_price_offering_strategy(
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
#     # If scenario profits are available, plot the cumulative distribution function

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
print(f"One-Price Expected Profit: {expected_profit:.2e} EUR")
print(f"Two-Price Expected Profit: {two_price_total_expected_profit:.2e} EUR")
print(f"Difference: {two_price_total_expected_profit - expected_profit:.2e} EUR")

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
print(f"One-Price Expected Profit: {expected_profit:.2e} EUR")
print(f"Two-Price Expected Profit: {two_price_total_expected_profit:.2e} EUR")
print(f"Expected Wind Profit: {ew_expected_profit:.2e} EUR")
print(f"Expected Wind vs One-Price: {ew_expected_profit - expected_profit:.2e} EUR")
print(f"Expected Wind vs Two-Price: {ew_expected_profit - two_price_total_expected_profit:.2e} EUR")

# Plot comparison of all three strategies
pf.compare_all_strategies(optimal_offers_one_price, optimal_offers_two_price, ew_optimal_offers)

print('#################################')
"""
# %% Ex-post Analysis

print("\n=== Ex-post Cross-validation Analysis ===")

# Combine in-sample and out-of-sample scenarios

# Perform cross-validation
# Replace current cross-validation call
results = s3.perform_cross_validation(
    in_sample_scenarios, 
    out_sample_scenarios,
    n_folds=8,
    capacity_wind_farm=CAPACITY_WIND_FARM,
    n_hours=N_HOURS
)

# Print results
print("\nCross-validation Results:")
print("========================")
for strategy in ['one_price', 'two_price']:
    print(f"\n{strategy.replace('_', ' ').title()} Strategy:")
    print(f"In-sample average profit: {results[strategy]['in_sample_avg']:.2e} ± {results[strategy]['in_sample_std']:.2e}")
    print(f"Out-sample average profit: {results[strategy]['out_sample_avg']:.2e} ± {results[strategy]['out_sample_std']:.2e}")

# Visualize results
plt.figure(figsize=(10, 6))
strategies = ['One-Price', 'Two-Price']
x = np.arange(len(strategies))
width = 0.35

in_sample_means = [results['one_price']['in_sample_avg'], results['two_price']['in_sample_avg']]
out_sample_means = [results['one_price']['out_sample_avg'], results['two_price']['out_sample_avg']]
in_sample_stds = [results['one_price']['in_sample_std'], results['two_price']['in_sample_std']]
out_sample_stds = [results['one_price']['out_sample_std'], results['two_price']['out_sample_std']]

plt.bar(x - width/2, in_sample_means, width, label='In-sample', 
        yerr=in_sample_stds, capsize=5, alpha=0.8)
plt.bar(x + width/2, out_sample_means, width, label='Out-of-sample', 
        yerr=out_sample_stds, capsize=5, alpha=0.8)

plt.xlabel('Strategy')
plt.ylabel('Expected Profit (EUR)')
plt.title('Cross-validation Results: In-sample vs Out-of-sample Profits')
plt.xticks(x, strategies)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/cross_validation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate percentage difference between in-sample and out-of-sample profits
for strategy in ['one_price', 'two_price']:
    in_sample = results[strategy]['in_sample_avg']
    out_sample = results[strategy]['out_sample_avg']
    diff_percent = ((in_sample - out_sample) / in_sample) * 100
    print(f"\n{strategy.replace('_', ' ').title()} Strategy Gap Analysis:")
    print(f"In-sample vs Out-sample difference: {diff_percent:.2e}%")
"""
print('#################################')

# %% Risk-Averse Offering Strategy
print("\n=== Risk-Averse Analysis (One-Price) ===")

from steps import step4_Risk_Averse as s4


# Define beta range from 0 (risk-neutral) to 1 (fully risk-averse)
beta_range_one_price = np.linspace(0, 1, 20)

risk_results = s4.analyze_risk_return_tradeoff(
    in_sample_scenarios=in_sample_scenarios,
    CAPACITY_WIND_FARM=CAPACITY_WIND_FARM,
    N_HOURS=N_HOURS,
    beta_values=beta_range_one_price
)

# Plot results 
plt.figure(figsize=(10, 6))
plt.plot(risk_results['cvar'], risk_results['expected_profit'], 'bo-', linewidth=2, markersize=6)

# Format axes
plt.xlabel("Conditional Value at Risk (CVaR) [kEUR]")
plt.ylabel("Expected Profit [kEUR]")
plt.title("Risk-Return Trade-off (One-Price, α = 0.90)")

# Set detailed y-axis ticks
y_min = min(risk_results['expected_profit'])
y_max = max(risk_results['expected_profit'])
y_range = y_max - y_min
plt.gca().yaxis.set_major_locator(plt.LinearLocator(10))  # Reduced number of ticks
plt.gca().yaxis.set_minor_locator(plt.LinearLocator(20))  # Reduced number of minor ticks

# Format tick labels with more precision
plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}'))
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.2f}'))
# Add grid for both major and minor ticks
plt.grid(True, which='major', alpha=0.3, linestyle='--')
plt.grid(True, which='minor', alpha=0.1, linestyle=':')

plt.tight_layout()
plt.savefig('results/risk_return_tradeoff.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table with improved formatting
print("\nRisk-Return Trade-off Analysis")
print("-" * 65)
print(f"{'Beta':^6} | {'Expected Profit':^15} | {'CVaR':^15} | {'Risk Premium':^15}")
print("-" * 65)

# Calculate and display risk premium
base_profit = risk_results['expected_profit'][0]  # Risk-neutral profit
for i, beta in enumerate(risk_results['beta']):
    exp_profit = risk_results['expected_profit'][i]
    cvar = risk_results['cvar'][i]
    risk_premium = base_profit - exp_profit
    
    print(f"{beta:6.2f} | {exp_profit/1000:15.2f} | {cvar/1000:15.2f} | {risk_premium/1000:15.2f}")
print("-" * 65)

# Additional analysis: Plot profit distribution for selected beta values
plt.figure(figsize=(12, 6))
selected_betas = [0.0, 0.5, 1.0]
for beta in selected_betas:
    idx = int(beta * 10)
    profits = list(risk_results['scenario_profits'][idx].values())
    plt.hist(profits, bins=30, alpha=0.5, label=f'β={beta:.1f}')

plt.xlabel('Profit [EUR]')
plt.ylabel('Number of Scenarios')
plt.title('Profit Distribution for Different Risk Levels')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/risk_profit_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Risk-Averse Analysis (Two-Price) ===")

beta_range_two_price = np.linspace(0, 1, 21)  # 11 unique values from 0 to 1

# Analyze two-price risk-return tradeoff
two_price_risk_results = s4.analyze_two_price_risk_return_tradeoff(
    in_sample_scenarios=in_sample_scenarios,
    CAPACITY_WIND_FARM=CAPACITY_WIND_FARM,
    N_HOURS=N_HOURS,
    beta_values=beta_range_two_price
)

# Plot two-price risk-return tradeoff
plt.figure(figsize=(10, 6))
plt.plot(two_price_risk_results['cvar'], two_price_risk_results['expected_profit'], 
         'ro-', linewidth=2, markersize=6)

# Format axes
plt.xlabel("Conditional Value at Risk (CVaR) [kEUR]")
plt.ylabel("Expected Profit [kEUR]")
plt.title("Risk-Return Trade-off (Two-Price, α = 0.90)")

# Set detailed y-axis ticks
y_min = min(two_price_risk_results['expected_profit'])
y_max = max(two_price_risk_results['expected_profit'])
y_range = y_max - y_min
plt.gca().yaxis.set_major_locator(plt.LinearLocator(10))  # Reduced number of ticks
plt.gca().yaxis.set_minor_locator(plt.LinearLocator(20))  # Minor ticks

# Format tick labels with more precision
plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}'))
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.2f}'))

# Add grid for both major and minor ticks
plt.grid(True, which='major', alpha=0.3, linestyle='--')
plt.grid(True, which='minor', alpha=0.1, linestyle=':')

plt.tight_layout()
plt.savefig('results/risk_return_tradeoff_two_price.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot profit distribution for two-price scheme
plt.figure(figsize=(12, 6))
selected_betas = [0.0, 0.5, 1.0]
n_points = len(two_price_risk_results['beta'])

for beta in selected_betas:
    # Calculate correct index based on number of points
    idx = int((n_points - 1) * beta)
    profits = list(two_price_risk_results['scenario_profits'][idx].values())
    plt.hist(profits, bins=30, alpha=0.5, label=f'β={beta:.1f}')

plt.xlabel('Profit [kEUR]')
plt.ylabel('Number of Scenarios')
plt.title('Profit Distribution for Different Risk Levels (Two-Price)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/risk_profit_distributions_two_price.png', dpi=300, bbox_inches='tight')
plt.show()

print('#################################')

# %% In-sample Decision Making: Offering Strategy Under the P90 Requirement
print("\n=== Offering Strategy Under the P90 Requirement ===")




print('#################################')

# %% Verification of the P90 Requirement Using Out-of-Sample Analysis
print("\n=== Verification of the P90 Requirement Using Out-of-Sample Analysis ===")





