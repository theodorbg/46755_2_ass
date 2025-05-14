print('\nImporting modules for main_step3.py')
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

# %% Risk-Averse Offering Strategy
print('#################################')
print('\nInitializing step4.py Risk-Averse Analysis (One-Price)')

CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0
N_HOURS = in_sample_scenarios[0].shape[0]  # 24 hours



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
s4.plot_risk_return_tradeoff(risk_results)

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
s4.plot_profit_distribution(scenario_profits, beta)

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
s4.plot_risk_return_tradeoff_two_price(risk_results)

# Plot profit distribution for two-price scheme
s4.plot_profit_distribution_two_price(two_price_scenario_profits, beta)


print('\nFinished step4.py: Ex-post Cross-validation Analysis')
print('#################################')

