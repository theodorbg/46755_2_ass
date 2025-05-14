print('#################################')
print('\nInitializing step2.py...')

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
import utils.step1_utils as s1
import utils.step2_utils as s2
from generate_consumption_profiles import in_sample_profiles, out_sample_profiles
from step1 import r_cvar_dtu, r_alsox_binary  # Importing the reserve bids from step1

# %% Verification of the P90 Requirement Using Out-of-Sample Analysis:
# The P90 requirement states that the reserve capacity offered should be available 
# at least 90% of the time. This section verifies if our previously calculated
# reserve bids (r_cvar_dtu and r_alsox_binary) satisfy this requirement.
print("\n=== Computing Verification of the P90 Requirement Using Out-of-Sample Analysis...")

# Extract reserve bids from step1
# reserve_cvar: Reserve bid calculated using Conditional Value-at-Risk approach
# reserve_alsox: Reserve bid calculated using ALSO-X binary approach
reserve_cvar = r_cvar_dtu
reserve_alsox = r_alsox_binary

# Run P90 verification
# Each call to verify_p90() checks if the reserve bid meets the P90 requirement
# by analyzing what percentage of time the consumption profile allows for the 
# specified reserve capacity reductions2.verify_p90(in_sample_profiles, reserve_cvar, "CVaR (In-sample)")
s2.verify_p90(out_sample_profiles, reserve_cvar, "CVaR (Out-of-sample)")
s2.verify_p90(in_sample_profiles, reserve_alsox, "ALSO-X (In-sample)")
s2.verify_p90(out_sample_profiles, reserve_alsox, "ALSO-X (Out-of-sample)")


# Compute shortfalls (in-sample)
# Shortfalls represent instances where the consumption profile doesn't provide
# enough flexibility to meet the committed reserve bid.
# For each case, we calculate:
# 1. When shortfalls occur (time points)
# 2. How large the shortfalls are (magnitude)
# 3. Distribution of shortfalls across different profiles
shortfalls_cvar_in = s2.compute_shortfalls(in_sample_profiles, reserve_cvar, "CVaR (In-sample)")
shortfalls_alsox_in = s2.compute_shortfalls(in_sample_profiles, reserve_alsox, "ALSO-X (In-sample)")

# Compute shortfalls (out-of-sample)
# This tests how well our reserve bids perform on previously unseen data (test set)
# Out-of-sample performance is crucial for evaluating the robustness of our solutionshortfalls_cvar_out = s2.compute_shortfalls(out_sample_profiles, reserve_cvar, "CVaR (Out-of-sample)")
shortfalls_alsox_out = s2.compute_shortfalls(out_sample_profiles, reserve_alsox, "ALSO-X (Out-of-sample)")

print("\n=== Verification of the P90 Requirement Using Out-of-Sample Analysis Computed ===")