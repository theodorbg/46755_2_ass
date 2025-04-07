import os
import numpy as np
import pandas as pd

import scenario_generation_functions as sgf

def generate_power_system_conditions(n_scenarios=4, hours=24, deficit_prob=0.5, random_seed=None):
    """
    Generate binary scenarios representing real-time power system conditions.
    
    Parameters:
        n_scenarios (int): Number of scenarios to generate.
        hours (int): Number of time periods (typically 24 for a full day).
        deficit_prob (float): Probability of power supply deficit (1 = deficit, 0 = excess).
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        pd.DataFrame: A dataframe with binary scenarios (1 = deficit, 0 = excess).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    scenarios = np.random.binomial(n=1, p=deficit_prob, size=(n_scenarios, hours))
    df = pd.DataFrame(scenarios, columns=[f"Hour_{i+1}" for i in range(hours)])
    df.index.name = "Scenario_ID"
    return df

def generate_balancing_prices(day_ahead_df, power_conditions_df):
    """
    Generate balancing price forecasts based on day-ahead prices and system conditions.
    
    Parameters:
        day_ahead_df (pd.DataFrame): Day-ahead price scenarios
        power_conditions_df (pd.DataFrame): Binary system conditions (1 = deficit, 0 = excess)
    
    Returns:
        pd.DataFrame: Balancing price scenarios
    """
    coefficients = np.where(power_conditions_df == 1, 1.25, 0.85)
    balancing_prices = day_ahead_df * coefficients
    return balancing_prices