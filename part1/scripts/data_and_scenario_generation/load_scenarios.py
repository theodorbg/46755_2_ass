# Standard library imports
import os
import pickle

# Third-party imports
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load scenarios from pickle files
def load_scenarios():
    """Load in-sample and out-of-sample scenarios from pickle files."""
    # Define paths to pickle files
    in_sample_path = os.path.join('part1', 'results', 'scenarios', 'in_sample_scenarios.pkl')
    out_sample_path = os.path.join('part1', 'results', 'scenarios', 'out_of_sample_scenarios.pkl')
    
    # Load in-sample scenarios
    with open(in_sample_path, 'rb') as f:
        in_sample_scenarios = pickle.load(f)
    
    # Load out-of-sample scenarios
    with open(out_sample_path, 'rb') as f:
        out_sample_scenarios = pickle.load(f)
    
    print(f"\nLoaded {len(in_sample_scenarios)} in-sample scenarios")
    print(f"Loaded {len(out_sample_scenarios)} out-of-sample scenarios")
    
    return in_sample_scenarios, out_sample_scenarios


