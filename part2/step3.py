print('#################################')
print('\nInitializing step3.py...')
#%% Imports
import random
from gurobipy import Model, GRB
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import utils.plot_functions as pf  # Importing the plot functions
import os
from generation.ConsumptionProfile import ConsumptionProfile  # Consumption profile generation class
import utils.step3_utils as s3
from generate_consumption_profiles import in_sample_profiles, out_sample_profiles

#%% Trade-off between reserve bid and expected shortfall
print("\n=== Initializing Energinets Perspective...")

# Define epsilon values (from P100 to P80)
# Epsilon represents the allowed violation probability (1-epsilon = reliability level)
# P100 means 100% reliability (no violations allowed), P80 means 80% reliability
MIN_VIOLATIONS = 0.0  # Minimum allowed violation
MAX_VIOLATIONS = 0.2  # Maximum allowed violation (20%)
epsilons = np.linspace(MIN_VIOLATIONS, MAX_VIOLATIONS, 21)  # 0% to 20% violation
p_requirements = 1 - epsilons

# Lists to store results for each epsilon value
reserve_bids = []          # Will store optimal reserve bids for each reliability level
expected_shortfalls = []   # Will store corresponding expected shortfalls


# Prepare data matrices from the consumption profiles
# in_matrix: Used for model training/optimization (in-sample)
# out_matrix: Used for performance evaluation (out-of-sample)in_matrix = np.array([
in_matrix, out_matrix, num_in_profiles, num_minutes, num_out_profiles = (
    s3.prepare_data_matrices(in_sample_profiles, out_sample_profiles))

# Loop over each epsilon (reliability level)
for epsilon in epsilons:
    reserve, expected_shortfall, reserve_bids, expected_shortfalls = (
        s3.solve_step3(epsilon, num_in_profiles, num_minutes, in_matrix,
                out_matrix, reserve_bids, expected_shortfalls))
    
# Plot trade-off between reserve bid and expected shortfall
s3.plot_tradeoff(reserve_bids, expected_shortfalls, p_requirements)


print("\n=== Computed Energinets Perspective ===")
print("\n=== Finished Part 2 ===")
