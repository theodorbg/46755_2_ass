# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to ensure directories exist before saving files
def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def plot_optimal_offers(offers, title="Optimal Day-Ahead Market Offers", filename=None):
    """Plot the optimal offers with customizable title and filename."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(24), offers)
    plt.xlabel('Hour')
    plt.ylabel('Offer Quantity (MW)')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    save_path = 'results/figures/solutions/one_price_optimal_offers.png'
    if filename:
        save_path = f'results/{filename}'
    
    # Ensure directory exists before saving
    ensure_dir_exists(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
def plot_cumulative_distribution_func(scenario_profits, expected_profit, scheme_type):
    profit_values = list(scenario_profits.values())
    plt.figure(figsize=(12, 6))

    sorted_profits = np.sort(profit_values)
    p = 1. * np.arange(len(sorted_profits)) / (len(sorted_profits) - 1)
    plt.plot(sorted_profits, p)

    plt.axvline(x=expected_profit, color='r', linestyle='--', label=f'Expected Profit: {expected_profit:.2f} EUR')
    plt.xlabel('Profit (EUR)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'Cumulative Distribution of Profit - {scheme_type} Balancing Scheme')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f'results/figures/solutions/{scheme_type}_profit_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
