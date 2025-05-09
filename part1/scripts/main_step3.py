print('#################################')
print('\nInitializing step3.py...')
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
plt.savefig('part1/results/step3/figures/cross_validation_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate percentage difference between in-sample and out-of-sample profits
for strategy in ['one_price', 'two_price']:
    in_sample = results[strategy]['in_sample_avg']
    out_sample = results[strategy]['out_sample_avg']
    diff_percent = ((in_sample - out_sample) / in_sample) * 100
    print(f"\n{strategy.replace('_', ' ').title()} Strategy Gap Analysis:")
    print(f"In-sample vs Out-sample difference: {diff_percent:.2e}%")

print('#################################')
