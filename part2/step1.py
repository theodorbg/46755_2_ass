# --- Script initialization ---
print('#################################')
print('\nInitializing step1.py...')

#%% --- Import necessary libraries ---
import random                                 # For random number generation
from gurobipy import Model, GRB              # Mathematical optimization solver
import numpy as np                           # Numerical computing library
import matplotlib.pyplot as plt              # Plotting library
import numpy as np                           # Duplicate import (could be removed)
import sys                                   # System-specific parameters and functions
from pathlib import Path                     # Object-oriented filesystem paths
import utils.plot_functions as pf            # Custom plotting utilities
import os                                    # Operating system interfaces
from generation.ConsumptionProfile import ConsumptionProfile  # Custom class for consumption profiles
import utils.step1_utils as s1               # Utility functions for Step 1 calculations
from generate_consumption_profiles import in_sample_profiles, out_sample_profiles  # Pre-generated consumption profiles

# --- PART 1: CVaR (Conditional Value at Risk) Approach ---
print("\n=== Computing CVaR (P90)...")

# Set reliability level parameter
epsilon = 0.1  # Corresponds to P90 reliability (90% confidence level, 10% risk level)

# Convert profile objects to numerical matrix for mathematical processing
# This creates a 2D array where each row is a consumption profile and each column is a minute
profiles_matrix = np.array([p.profile for p in in_sample_profiles])  # shape: (num_profiles, 60)

# Extract dimensions of the data
num_profiles, num_minutes = profiles_matrix.shape  # e.g., (200, 60) for 200 profiles, 60 minutes each

# Calculate total number of data points for violation counting
N = num_profiles * num_minutes  # Total number of consumption values across all profiles

# Solve the CVaR optimization model
# This determines the maximum reserve capacity that can be reliably offered
r_cvar_dtu, betaX, zeta = s1.solve_step1(profiles_matrix, num_profiles, num_minutes, N, epsilon)

# Display the results
print(f"Optimal reserve capacity bid (CVaR) under P90: {r_cvar_dtu:.2f} kW")
print("\n=== CVaR (P90) Computed ===")

# --- PART 2: ALSO-X (Allowance of Limited Shortfalls) Approach ---
print("\n=== Computing ALSO - X...")

# Calculate maximum allowed violations based on the reliability level
max_violations = int(epsilon * num_profiles * num_minutes)  # 10% of all data points can be violated
print(f"Max violations allowed: {max_violations} (10% of {num_profiles * num_minutes})")

# Big-M parameter for the mixed-integer programming formulation
# This is a large constant used in the constraint formulation
M = 10000  # Sufficiently large value to enforce logical constraints

# Solve the ALSO-X model using Mixed Integer Linear Programming (MILP)
r_alsox_binary = s1.solve_milp_model(profiles_matrix, num_profiles,
                                   num_minutes, max_violations, M)

# Visualize the results by creating a flexibility plot
# This shows both reserve capacity bids in the context of the consumption profiles
s1.create_flexibility_plot(profiles=in_sample_profiles, r_cvar=r_cvar_dtu,
                         r_alsox=r_alsox_binary, max_power=600,
                         show_plot=False  # Save plot to file instead of displaying
                         )

# Display the ALSO-X results
print(f"\nOptimal reserve capacity bid (ALSO-X MILP) under P90: {r_alsox_binary:.2f} kW")
print("\n=== ALSO-X Computed ===")

# --- Script completion ---
print('\nStep1.py completed')
print('#################################')