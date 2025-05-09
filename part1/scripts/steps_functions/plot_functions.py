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
    save_path = 'part1/results/step1/figures/one_price_optimal_offers.png'
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
    plt.savefig(f'part1/results/step1/figures/{scheme_type}_profit_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_offers(optimal_offers_one_price, optimal_offers_two_price):
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(24) - 0.2, optimal_offers_one_price, width=0.4, label='One-Price', color='blue', alpha=0.7)
    plt.bar(np.arange(24) + 0.2, optimal_offers_two_price, width=0.4, label='Two-Price', color='orange', alpha=0.7)
    plt.xlabel('Hour')
    plt.ylabel('Offer Quantity (MW)')
    plt.title('Comparison of Optimal Day-Ahead Offers')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('part1/results/step2/figures/comparison_optimal_offers.png', dpi=300, bbox_inches='tight')
    print('\nPlotted comparison of offering strategies and saved to results/comparison_optimal_offers.png')
    plt.close()

def compare_all_strategies(optimal_offers_one_price, optimal_offers_two_price, ew_optimal_offers):
    plt.figure(figsize=(14, 7))
    plt.bar(np.arange(24) - 0.3, optimal_offers_one_price, width=0.2, label='One-Price', color='blue', alpha=0.7)
    plt.bar(np.arange(24), optimal_offers_two_price, width=0.2, label='Two-Price', color='green', alpha=0.7)
    plt.bar(np.arange(24) + 0.3, ew_optimal_offers, width=0.2, label='Expected Wind', color='orange', alpha=0.7)
    plt.xlabel('Hour')
    plt.ylabel('Offer Quantity (MW)')
    plt.title('Comparison of Bidding Strategies')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('part1/results/step2/figures/comparison_all_strategies.png', dpi=300, bbox_inches='tight')

