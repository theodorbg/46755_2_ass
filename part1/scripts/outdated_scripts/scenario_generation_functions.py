import os
import numpy as np
import pandas as pd

import scripts.outdated_scripts.scenario_generation_functions as sgf


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