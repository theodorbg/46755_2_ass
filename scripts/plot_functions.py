# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_optimal_offers(optimal_offers):
    plt.figure(figsize=(12, 6))
    plt.bar(range(24), optimal_offers)
    plt.xlabel('Hour')
    plt.ylabel('Offer Quantity (MW)')
    plt.title('Optimal Day-Ahead Market Offers')
    plt.grid(alpha=0.3)
    plt.savefig('results//figures/1_1/one_price_optimal_offers.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_distribution_func(scenario_profits, expected_profit):
    profit_values = list(scenario_profits.values())
    plt.figure(figsize=(12, 6))

    sorted_profits = np.sort(profit_values)
    p = 1. * np.arange(len(sorted_profits)) / (len(sorted_profits) - 1)
    plt.plot(sorted_profits, p)

    plt.axvline(x=expected_profit, color='r', linestyle='--', label=f'Expected Profit: {expected_profit:.2f} EUR')
    plt.xlabel('Profit (EUR)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Profit - One-Price Balancing Scheme')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('results/figures/1_1/one_price_profit_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
