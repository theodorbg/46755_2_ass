print('#################################')
print('\nInitializing step4.py...')
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

# Read in_sample and out of_sample scenarios
in_sample_scenarios, out_sample_scenarios = load_scenarios.load_scenarios()

CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0
N_HOURS = in_sample_scenarios[0].shape[0]  # 24 hours
# %% Risk-Averse Offering Strategy
print("\n=== Risk-Averse Analysis (One-Price) ===")

from steps_functions import step4_Risk_Averse as s4


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
plt.savefig('part1/results/step4/figures/risk_return_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

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
plt.savefig('part1/results/step4/figures/risk_profit_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

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
plt.savefig('part1/results/step4/figures/risk_return_tradeoff_two_price.png', dpi=300, bbox_inches='tight')
plt.close()

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
plt.savefig('part1/results/step4/figures/risk_profit_distributions_two_price.png', dpi=300, bbox_inches='tight')
plt.close()
