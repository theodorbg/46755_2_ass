# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_scenario(scenario_df, scenario_id):
    """
    Plot all data for a specific scenario
    
    Args:
        scenario_df: DataFrame containing the scenario data
        scenario_id: ID of the scenario (for the title)
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot condition
    axes[0].step(range(24), scenario_df['condition'], where='post', lw=2)
    axes[0].set_ylabel('Condition\n(0=Excess, 1=Deficit)')
    axes[0].set_title(f'Scenario {scenario_id}', fontsize=16)
    axes[0].grid(True, alpha=0.3)
    
    # Plot prices
    axes[1].plot(range(24), scenario_df['price'], 'b-', label='Price')
    axes[1].plot(range(24), scenario_df['balancing_price'], 'r--', label='Balancing Price')
    axes[1].set_ylabel('Price (EUR/MWh)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot wind
    axes[2].plot(range(24), scenario_df['wind'], 'g-')
    axes[2].set_ylabel('Wind (MW)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot imbalance cost (difference between price and balancing price)
    imbalance_cost = scenario_df['balancing_price'] - scenario_df['price']
    axes[3].bar(range(24), imbalance_cost, color=['red' if x > 0 else 'green' for x in imbalance_cost])
    axes[3].set_ylabel('Imbalance Cost\n(EUR/MWh)')
    axes[3].set_xlabel('Hour')
    axes[3].grid(True, alpha=0.3)
    
    # Format x-axis
    axes[3].set_xticks(range(0, 24, 2))
    
    # Save figure
    os.makedirs('results/figures/scenarios/example_visualizations', exist_ok=True)
    plt.savefig(f'results/figures/scenarios/example_visualizations{scenario_id}.png', 
                dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

def plot_balancing_prices_by_day(balancing_prices_list, df_price,
                                 save_dir='results/figures/balancing_prices'):
    """
    Create 20 plots (one for each day) showing all four condition-specific balancing prices
    and the original price for that day.
    
    Args:
        balancing_prices_list: List of balancing price DataFrames, one for each condition
        df_price: Original price DataFrame
        save_dir: Directory to save the plots
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Number of days
    days = df_price.shape[1]  # 20 days
    
    # Iterate through each day
    for day_idx in range(days):
        plt.figure(figsize=(12, 6))
        
        # Plot the original price for this day
        plt.plot(df_price.index, df_price.iloc[:, day_idx], 
                 label='Day-Ahead Price', linewidth=2.5, linestyle = '--')
        
        # Colors for different conditions
        colors = ['blue', 'red', 'green', 'purple']
        
        # Plot all four balancing prices for this day
        for condition in range(len(balancing_prices_list)):
            plt.plot(
                balancing_prices_list[condition].index, 
                balancing_prices_list[condition].iloc[:, day_idx],
                color=colors[condition],
                label=f'Condition {condition}',
                linewidth=1.5
                # linestyle=['--', ':', '-.', '-'][condition % 4]
            )
        
        # Add title and labels
        plt.title(f'Day {day_idx+1}: Price and Balancing Prices for All Conditions', fontsize=14)
        plt.xlabel('Hour', fontsize=12)
        plt.ylabel('Price (EUR/MWh)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Add annotation explaining conditions
        plt.annotate(
            'Condition 0: Excess (+25%)\nCondition 1: Deficit (-15%)', 
            xy=(0.02, 0.02), xycoords='axes fraction',
            fontsize=9, bbox=dict(facecolor='white', alpha=0.7)
        )
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'{save_dir}/day_{day_idx+1}_all_conditions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Generated {days} plots, one for each day, showing all balancing price conditions")
