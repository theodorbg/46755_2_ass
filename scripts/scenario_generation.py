import numpy as np
import pandas as pd

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

# Example usage
n_scenarios = 4  # Number of system condition scenarios (you'll later combine with others to reach 1600)
deficit_probability = 0.5  # 50% chance of deficit or excess

power_condition_df = generate_power_system_conditions(n_scenarios=n_scenarios, deficit_prob=deficit_probability, random_seed=42)

print(power_condition_df)