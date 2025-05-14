print('#################################')
print('\nInitializing step1.py...')
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
import utils.step1_utils as s1
from generate_consumption_profiles import in_sample_profiles, out_sample_profiles

# %% Data generation
# %% CVaR
print("\n=== Computing CVaR (P90)...")

# Parameters for the CVaR model
epsilon = 0.1  # Confidence level for CVaR (1 - alpha). Here, alpha = 0.9 (P90), so epsilon = 0.1.

# Convert list of ConsumptionProfile objects into a 2D NumPy array.
# Each row is a profile, each column is a minute's consumption.
profiles_matrix = np.array([p.profile for p in in_sample_profiles])  # shape: (num_profiles, 60)

# Get dimensions: number of profiles and minutes per profile.
num_profiles, num_minutes = profiles_matrix.shape

# Total number of data points (profile-minute combinations).
N = num_profiles * num_minutes  

# Solve the model
r_cvar_dtu, betaX, zeta = s1.solve_step1(profiles_matrix, num_profiles, num_minutes, N, epsilon)


# Output the results
print(f"Optimal reserve capacity bid (CVaR) under P90: {r_cvar_dtu:.2f} kW")
print("\n=== CVaR (P90) Computed ===")
# %% ALSO - X
print("\n=== Computing ALSO - X...")
max_violations = int(epsilon * num_profiles * num_minutes)
print(f"Max violations allowed: {max_violations} (10% of {num_profiles * num_minutes})")
M = 10000  # Big-M constant


r_alsox_binary = s1.solve_milp_model(profiles_matrix, num_profiles,
                                     num_minutes, max_violations, M)

# Plotting
s1.create_flexibility_plot(profiles=in_sample_profiles, r_cvar=r_cvar_dtu,
                           r_alsox=r_alsox_binary, max_power=600,
                           show_plot=False
                           )
print(f"\nOptimal reserve capacity bid (ALSO-X MILP) under P90: {r_alsox_binary:.2f} kW")
print("\n=== ALSO-X Computed ===")

print('\nStep1.py completed')
print('#################################')
