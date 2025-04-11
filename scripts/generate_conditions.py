import numpy as np
import pandas as pd

def generate_conditions(n_scenarios=4, deficit_prob=0.5, random_seed=42):
    """
    Generate a DataFrame of power system conditions (0=excess, 1=deficit).
    
    Returns:
        DataFrame with hours as rows and scenarios as columns
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate binary scenarios (1=deficit, 0=excess)
    scenarios = np.random.binomial(n=1, p=deficit_prob, size=(n_scenarios, 24))
    
    # Create and format DataFrame
    df_conditions = pd.DataFrame(
        data=scenarios.T,  # Transpose to have hours as rows
        index=pd.Index(range(24), name='Hour'),
        columns=[f'Scenario {i+1}' for i in range(n_scenarios)]
    )
    
    return df_conditions


# Generate and display the conditions DataFrame
df_conditions = generate_conditions()
# print(df_conditions)