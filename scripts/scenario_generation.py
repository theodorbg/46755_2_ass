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

if __name__ == "__main__":
    # Generate power system conditions
    n_scenarios = 4
    deficit_probability = 0.5
    power_conditions_df = generate_power_system_conditions(
        n_scenarios=n_scenarios, 
        deficit_prob=deficit_probability, 
        random_seed=42
    )
    
    # Load and prepare day-ahead prices from Excel
    day_ahead_prices_raw = pd.read_excel(
        'data/Price_dayahead.xlsx', 
        engine='openpyxl',
        decimal=',',  # Handle comma as decimal separator
        index_col=0
    )
    
    # Select all 20 days of data (reversing order to get day 1-20)
    all_days_prices = day_ahead_prices_raw.iloc[:, ::-1].values
    
    # Create scenarios for each day
    all_scenarios = []
    for day in range(20):  # Process each day
        day_prices = all_days_prices[:, day]
        # Reshape to match scenario format
        day_scenarios = np.tile(day_prices, (n_scenarios, 1))
        day_df = pd.DataFrame(
            day_scenarios,
            columns=[f"Hour_{i+1}" for i in range(24)],
            index=[f"Day_{day+1}_Scenario_{i}" for i in range(n_scenarios)]
        )
        all_scenarios.append(day_df)
    
    # Combine all scenarios
    day_ahead_df = pd.concat(all_scenarios)
    
    # Generate system conditions for all scenarios
    power_conditions_df = generate_power_system_conditions(
        n_scenarios=n_scenarios * 20,  # 4 scenarios × 20 days
        deficit_prob=deficit_probability,
        random_seed=42
    )
    
    # Generate balancing prices
    balancing_prices_df = generate_balancing_prices(day_ahead_df, power_conditions_df)
    
    # Print results
    print("\nPower System Conditions (1 = deficit, 0 = excess):")
    print(power_conditions_df)
    print("\nDay-Ahead Prices (showing first few rows):")
    print(day_ahead_df.head())
    print("\nBalancing Prices (showing first few rows):")
    print(balancing_prices_df.head())
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total number of scenarios: {len(day_ahead_df)}")
    print(f"Number of days: 20")
    print(f"Scenarios per day: {n_scenarios}")