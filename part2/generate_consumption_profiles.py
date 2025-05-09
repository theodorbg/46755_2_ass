print("\nInitializing main_part2.py")
import sys
from pathlib import Path
import random

# Add project root to Python path
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import plot_functions.plot_functions as pf  # Importing the plot functions
import os
from generation.ConsumptionProfile import ConsumptionProfile  # Consumption profile generation class
from generation.ConsumptionProfile import verify_profiles  # Verification function

print('\nGenerating consumption profiles...')
# Generate 300 random consumption profiles
consumption_profiles = [ConsumptionProfile(220, 600, 35, 1, 1) for _ in range(300)]

# Pick out 100 random profiles for in-sample and 200 for out-of-sample
in_sample_profiles = random.sample(consumption_profiles, 100)
out_sample_profiles = [profile for profile in consumption_profiles if profile not in in_sample_profiles]

# Verify the profiles
verify_profiles(in_sample_profiles, out_sample_profiles)

# Visualize the first 10 in-sample profiles
pf.plot_consumption_profiles(in_sample_profiles, out_sample_profiles)
