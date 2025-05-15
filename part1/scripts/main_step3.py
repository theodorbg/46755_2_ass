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
import data_and_scenario_generation.scenario_generator 

print('#################################')
print('\nInitializing step3.py: Ex-post Cross-validation Analysis ...')


# Read in_sample and out of_sample scenarios
in_sample_scenarios, out_sample_scenarios = load_scenarios.load_scenarios()

#Define constants
CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0
N_HOURS = in_sample_scenarios[0].shape[0]  # 24 hours
# %% Ex-post Analysis
# Combine in-sample and out-of-sample scenarios

# Perform cross-validation
results = s3.perform_cross_validation(in_sample_scenarios, 
                                      out_sample_scenarios,
                                      n_folds=8,
                                      capacity_wind_farm=CAPACITY_WIND_FARM,
                                      n_hours=N_HOURS)


# Print results
print("\nCross-validation Results:")
print("========================")
s3.print_strategy_results(results)

# Visualize results
in_sample_means, out_sample_means, in_sample_stds, out_sample_stds, strategies = s3.plot_cross_validation(results)

one_price_in, one_price_out, two_price_in, two_price_out, folds = s3.plot_fold_evolution(results)

s3.plot_combined_results(results, in_sample_means, out_sample_means,
                          in_sample_stds, out_sample_stds, strategies,
                          one_price_in, one_price_out, two_price_in, two_price_out, folds)

# Calculate percentage difference between in-sample and out-of-sample profits
s3.gap_analysis(results)

print('\nFinished step3.py: Ex-post Cross-validation Analysis')
print('#################################')
