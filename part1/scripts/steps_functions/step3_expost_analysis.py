import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .step1_one_price import solve_one_price_offering_strategy
from .step2_two_price import solve_two_price_offering_strategy
# from .step2_two_price import solve_two_price_offering_strategy_hourly

def calculate_profits(offers, scenarios, capacity_wind_farm, n_hours, price_scheme='one'):
    """
    Calculate the expected profit across multiple scenarios given day-ahead market offers.
    
    This function evaluates the economic performance of a set of hourly electricity market
    offers by calculating the total profit (day-ahead revenue + balancing settlement) across
    all provided scenarios and hours.
    
    Args:
        offers: List containing the day-ahead market offers (MW) for each hour
        scenarios: Dictionary of scenarios, each containing wind, price, balancing_price and condition data
        capacity_wind_farm: Maximum wind farm capacity in MW (not directly used in this function)
        n_hours: Number of hours in the planning horizon (typically 24)
        price_scheme: Balancing scheme type - 'one' for one-price scheme, anything else for two-price scheme
        
    Returns:
        float: Average profit across all scenarios (€)
    """
    total_profit = 0  # Initialize accumulator for total profit across all scenarios
    
    # Iterate through each scenario to calculate its profit
    for s in scenarios:
        scenario = scenarios[s]  # Get the data for this scenario
        scenario_profit = 0  # Initialize accumulator for this specific scenario
        
        # Calculate profit for each hour in this scenario
        for h in range(n_hours):
            # Extract relevant data from the scenario for this hour
            wind_actual = scenario.loc[h, 'wind']           # Actual wind production (MW)
            price_DA = scenario.loc[h, 'price']             # Day-ahead market price (€/MWh)
            price_BAL = scenario.loc[h, 'balancing_price']  # Balancing market price (€/MWh)
            condition = scenario.loc[h, 'condition']        # System condition (0=excess, 1=deficit)
            
            # --- Part 1: Day-ahead market revenue calculation ---
            p_DA = offers[h]                 # Day-ahead offer for this hour (MW)
            revenue_DA = price_DA * p_DA     # Day-ahead revenue (€)
            
            # --- Part 2: Balancing market settlement calculation ---
            imbalance = wind_actual - p_DA   # Imbalance: actual production minus offered amount (MW)
                                            # Positive: surplus (more wind than offered)
                                            # Negative: deficit (less wind than offered)
            
            # Calculate balancing settlement based on price scheme
            if price_scheme == 'one':
                # --- One-price balancing scheme ---
                # In one-price scheme, both surplus and deficit are settled at the same balancing price
                # revenue_BAL is positive for surplus (extra money received) and
                # negative for deficit (money paid back)
                revenue_BAL = price_BAL * imbalance
            else:
                # --- Two-price balancing scheme ---
                # In two-price scheme, the settlement price depends on both:
                # 1. The direction of the imbalance (surplus or deficit)
                # 2. The system condition (excess or deficit)
                
                if condition == 0:  # System excess condition
                    # - Positive imbalance (surplus): Paid at balancing price (typically < day-ahead price)
                    # - Negative imbalance (deficit): Charged at day-ahead price
                    revenue_BAL = (price_BAL * max(0, imbalance) +  # Revenue for surplus (if any)
                                  price_DA * min(0, imbalance))     # Cost for deficit (if any)
                else:  # System deficit condition (condition == 1)
                    # - Positive imbalance (surplus): Paid at day-ahead price
                    # - Negative imbalance (deficit): Charged at balancing price (typically > day-ahead price)
                    revenue_BAL = (price_DA * max(0, imbalance) +   # Revenue for surplus (if any)
                                  price_BAL * min(0, imbalance))    # Cost for deficit (if any)
            
            # Add this hour's profit (day-ahead revenue + balancing settlement) to the scenario total
            scenario_profit += revenue_DA + revenue_BAL
        
        # Add this scenario's profit to the total profit
        total_profit += scenario_profit
    
    # Return the average profit across all scenarios
    # This represents the expected profit under the given probability distribution
    return total_profit / len(scenarios)

def perform_cross_validation(in_sample_scenarios, out_sample_scenarios, n_folds=8, 
                           capacity_wind_farm=500, n_hours=24):
    """
    Perform k-fold cross-validation analysis using sequential blocks of scenarios.
    
    Args:
        in_sample_scenarios: Dictionary of original in-sample scenarios (keys 0-199)
        out_sample_scenarios: Dictionary of out-of-sample scenarios (keys 200+)
        n_folds: Number of cross-validation folds
        capacity_wind_farm: Maximum capacity of wind farm (MW)
        n_hours: Number of hours in planning horizon
        
    Returns:
        Dictionary with cross-validation results
    """
    # Initialize results dictionary
    results = {
        'one_price': {'in_sample': [], 'out_sample': []},
        'two_price': {'in_sample': [], 'out_sample': []}
    }
    
    # Calculate in_sample_size (number of scenarios in the initial in-sample set, e.g., 200)
    in_sample_size = len(in_sample_scenarios)
    
    # The variable all_keys is defined on line 67 but not used in the excerpt from line 70.
    # If not used later in the function, it can be removed.
    # all_keys = list(in_sample_scenarios.keys()) + list(out_sample_scenarios.keys())
    
    # Process each fold
    for fold in range(n_folds):
        print(f"Processing fold {fold + 1}/{n_folds}")
        
        if fold == 0:
            # First fold uses original in-sample scenarios (keys 0-199)
            fold_in_sample = in_sample_scenarios
            # And original out-of-sample scenarios (keys 200-1599)
            fold_out_sample = out_sample_scenarios
        else:
            # For subsequent folds, select a new block of scenarios as in-sample.
            # This block is taken from the original out_sample_scenarios.
            # The size of this block is determined by in_sample_size (e.g., 200).
            
            # Calculate start and end indices for slicing the list of out_sample_scenarios' keys
            start_idx = (fold - 1) * in_sample_size 
            end_idx = start_idx + in_sample_size
            
            # Ensure we don't go out of bounds for out_sample_scenarios
            if start_idx >= len(out_sample_scenarios):
                print(f"Warning: Not enough unique out-of-sample scenario blocks for fold {fold + 1}")
                break # Stop if no more blocks can be formed
                
            # Get the list of keys from the original out_sample_scenarios
            out_sample_keys_list = list(out_sample_scenarios.keys()) # e.g., [200, 201, ..., 1599]
            # Select a slice of these keys for the current fold's in-sample set
            fold_in_sample_keys = out_sample_keys_list[start_idx:min(end_idx, len(out_sample_keys_list))]
            
            # Create the in-sample dictionary for this fold using scenarios from original out_sample_scenarios
            fold_in_sample = {key: out_sample_scenarios[key] for key in fold_in_sample_keys}
            
            # Create the out-sample dictionary for this fold.
            # It will contain all original in-sample scenarios (0-199)
            # PLUS all original out-of-sample scenarios (200-1599) that are NOT in fold_in_sample_keys.
            fold_out_sample = {}
            
            # Add all original in-sample scenarios (keys 0-199)
            for k_orig_in in in_sample_scenarios:
                fold_out_sample[k_orig_in] = in_sample_scenarios[k_orig_in]
                
            # Add original out-of-sample scenarios (keys 200-1599) that are not part of the current fold_in_sample
            for k_orig_out in out_sample_scenarios:
                if k_orig_out not in fold_in_sample_keys: # fold_in_sample_keys are from out_sample_scenarios.keys()
                    fold_out_sample[k_orig_out] = out_sample_scenarios[k_orig_out]
            
            # The 'fold_out_sample_keys' variable previously defined on lines 89-90 was not directly
            # used for constructing 'fold_out_sample' dictionary above, so it can be omitted.
        
        print(f"  Fold {fold + 1}: In-sample size: {len(fold_in_sample)}, Out-sample size: {len(fold_out_sample)}")
        
        # Solve strategies and calculate profits
        for strategy, solver_func in [ # Renamed solver to solver_func to avoid conflict if solver is a module
            ('one_price', solve_one_price_offering_strategy),
            ('two_price', solve_two_price_offering_strategy) # Assuming this is not the hourly one for this main CV loop
        ]:
            # Note: If Gurobi size limits are an issue with solve_two_price_offering_strategy,
            # you might need to switch to solve_two_price_offering_strategy_hourly here too,
            # or ensure the number of scenarios in fold_in_sample is small enough.
            
            # Solve offering strategy
            # Assuming all solver functions now consistently return three values: offers, total_expected_profit, scenario_specific_profits
            optimal_offers, returned_total_profit, returned_scenario_profits = solver_func(fold_in_sample, capacity_wind_farm, n_hours)
            
            if optimal_offers is None: # Handle cases where solver might fail (e.g., non-optimal status)
                print(f"Warning: Solver for {strategy} failed for fold {fold + 1}. Skipping profit calculation for this strategy/fold.")
                # Append NaN or handle as appropriate for averaging later, or skip appending
                results[strategy]['in_sample'].append(np.nan) # Or 0, or skip
                results[strategy]['out_sample'].append(np.nan) # Or 0, or skip
                continue

            # Calculate profits using the dedicated calculate_profits function
            # The 'total_profit' and 'scenario_profits' from the solver might be based on its internal objective,
            # while calculate_profits provides a consistent evaluation metric.
            in_sample_profit_eval = calculate_profits(
                optimal_offers, fold_in_sample, capacity_wind_farm, n_hours, 
                strategy.split('_')[0] # 'one' or 'two'
            )
            out_sample_profit_eval = calculate_profits(
                optimal_offers, fold_out_sample, capacity_wind_farm, n_hours,
                strategy.split('_')[0] # 'one' or 'two'
            )
            
            # Store results
            results[strategy]['in_sample'].append(in_sample_profit_eval)
            results[strategy]['out_sample'].append(out_sample_profit_eval)
    
    # Calculate summary statistics (mean and std for profits collected from each fold)
    for strategy in results:
        for sample_type in ['in_sample', 'out_sample']:
            profits_list = results[strategy][sample_type]
            # Filter out NaNs if they were added, before calculating mean/std
            valid_profits = [p for p in profits_list if not np.isnan(p)]
            if valid_profits:  # Check if list is not empty after filtering
                results[strategy][f'{sample_type}_avg'] = np.mean(valid_profits)
                results[strategy][f'{sample_type}_std'] = np.std(valid_profits)
            else:
                results[strategy][f'{sample_type}_avg'] = np.nan # Or 0
                results[strategy][f'{sample_type}_std'] = np.nan # Or 0
    
    return results

def print_strategy_results(results):
    """Print the results of the strategies"""

    for strategy in ['one_price', 'two_price']:
        print(f"\n{strategy.replace('_', ' ').title()} Strategy:")
        print(f"In-sample average profit: {results[strategy]['in_sample_avg']:.2e} ± {results[strategy]['in_sample_std']:.2e}")
        print(f"Out-sample average profit: {results[strategy]['out_sample_avg']:.2e} ± {results[strategy]['out_sample_std']:.2e}")

    return None

def plot_cross_validation(results):
    """Plot the cross-validation results for in-sample and out-of-sample profits"""
    
    plt.figure(figsize=(10, 6))
    strategies = ['One-Price', 'Two-Price']
    x = np.arange(len(strategies))
    width = 0.35

    in_sample_means = [results['one_price']['in_sample_avg'], results['two_price']['in_sample_avg']]
    out_sample_means = [results['one_price']['out_sample_avg'], results['two_price']['out_sample_avg']]
    in_sample_stds = [results['one_price']['in_sample_std'], results['two_price']['in_sample_std']]
    out_sample_stds = [results['one_price']['out_sample_std'], results['two_price']['out_sample_std']]

    # Create bars with custom colors and better styling
    plt.bar(x - width/2, in_sample_means, width, label='In-sample', 
            color='#2E86C1', yerr=in_sample_stds, capsize=5, 
            alpha=0.8, error_kw={'ecolor': '0.2', 'capthick': 2})
    plt.bar(x + width/2, out_sample_means, width, label='Out-of-sample', 
            color='#E67E22', yerr=out_sample_stds, capsize=5, 
            alpha=0.8, error_kw={'ecolor': '0.2', 'capthick': 2})

    # Set y-axis limits to start from 300,000
    plt.ylim(bottom=300000)  # Add this line

    plt.xlabel('Strategy')
    plt.ylabel('Expected Profit (EUR)')
    plt.title('Cross-validation Results: In-sample vs Out-of-sample Profits')
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part1/results/step3/figures/cross_validation_results.png', dpi=300, bbox_inches='tight')
    print('\nPlotted cross-validation results and saved to part1/results/step3/figures/cross_validation_results.png')
    plt.close()

    return in_sample_means, out_sample_means, in_sample_stds, out_sample_stds, strategies


def plot_fold_evolution(results):
    """Create a visualization of profit evolution across folds"""
    
    # Create figure with appropriate size for a single plot
    plt.figure(figsize=(10, 5))
    
    # Prepare data for fold comparison
    one_price_in = results['one_price']['in_sample']
    one_price_out = results['one_price']['out_sample']
    two_price_in = results['two_price']['in_sample']
    two_price_out = results['two_price']['out_sample']
    
    folds = range(1, len(one_price_in) + 1)
    
    # Calculate means for each series
    one_price_in_mean = np.mean(one_price_in)
    one_price_out_mean = np.mean(one_price_out)
    two_price_in_mean = np.mean(two_price_in)
    two_price_out_mean = np.mean(two_price_out)

    # print results
    print(f"\nFold-by-fold results:")
    print(f"One-Price In-sample: {one_price_in}")
    print(f"One-Price Out-of-sample: {one_price_out}")
    print(f"Two-Price In-sample: {two_price_in}")
    print(f"Two-Price Out-of-sample: {two_price_out}")
    
    print(f"One-Price In-sample mean: {one_price_in_mean:.2e} EUR")
    print(f"One-Price Out-of-sample mean: {one_price_out_mean:.2e} EUR")
    print(f"Two-Price In-sample mean: {two_price_in_mean:.2e} EUR")
    print(f"Two-Price Out-of-sample mean: {two_price_out_mean:.2e} EUR")
    print(f"Fold-by-fold results saved to part1/results/step3/fold_evolution.txt")
    with open('part1/results/step3/fold_evolution.txt', 'w') as f:

        f.write(f"Fold-by-fold results:\n")
        f.write(f"One-Price In-sample: {one_price_in}\n")
        f.write(f"One-Price Out-of-sample: {one_price_out}\n")
        f.write(f"Two-Price In-sample: {two_price_in}\n")
        f.write(f"Two-Price Out-of-sample: {two_price_out}\n")
        f.write(f"One-Price In-sample mean: {one_price_in_mean:.2e} EUR\n")
        f.write(f"One-Price Out-of-sample mean: {one_price_out_mean:.2e} EUR\n")
        f.write(f"Two-Price In-sample mean: {two_price_in_mean:.2e} EUR\n")
        f.write(f"Two-Price Out-of-sample mean: {two_price_out_mean:.2e} EUR\n")
        f.write(f"Fold-by-fold results saved to part1/results/step3/fold_evolution.txt\n")



    # Plot fold-by-fold comparison with enhanced styling
    plt.plot(folds, one_price_in, 'o-', color='#2E86C1', label='One-Price In-sample', 
             alpha=0.9, linewidth=2.5, markersize=8)
    plt.plot(folds, one_price_out, 's-', color='#2E86C1', label='One-Price Out-of-sample', 
             linestyle='--', alpha=0.9, linewidth=2.5, markersize=8)
    plt.plot(folds, two_price_in, 'o-', color='#E67E22', label='Two-Price In-sample', 
             alpha=0.9, linewidth=2.5, markersize=8)
    plt.plot(folds, two_price_out, 's-', color='#E67E22', label='Two-Price Out-of-sample',
             linestyle='--', alpha=0.9, linewidth=2.5, markersize=8)
    
    # Add mean lines
    plt.axhline(y=one_price_in_mean, color='#2E86C1', linestyle='-', alpha=0.3)
    plt.axhline(y=one_price_out_mean, color='#2E86C1', linestyle='--', alpha=0.3)
    plt.axhline(y=two_price_in_mean, color='#E67E22', linestyle='-', alpha=0.3)
    plt.axhline(y=two_price_out_mean, color='#E67E22', linestyle='--', alpha=0.3)

    # Customize plot
    plt.xlabel('Fold Number', fontsize=14, fontweight='bold')
    plt.ylabel('Expected Profit (EUR)', fontsize=14, fontweight='bold')
    plt.title('Cross-validation Results: Profit Evolution Across Folds', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Move legend inside the plot
    plt.legend(fontsize=12, ncol=2, loc='lower right', 
              bbox_to_anchor=(0.98, 0.02))
    
    # Customize ticks
    plt.xticks(folds, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Format y-axis with comma separator
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('part1/results/step3/figures/fold_evolution.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print('\nPlotted fold evolution results and saved to part1/results/step3/figures/fold_evolution.png')
    plt.close()

    return one_price_in, one_price_out, two_price_in, two_price_out, folds

def  gap_analysis(results):
    ''' Calculate percentage difference between in-sample and out-of-sample profits'''
    for strategy in ['one_price', 'two_price']:
        in_sample = results[strategy]['in_sample_avg']
        out_sample = results[strategy]['out_sample_avg']
        diff_percent = ((in_sample - out_sample) / in_sample) * 100
        print(f"\n{strategy.replace('_', ' ').title()} Strategy Gap Analysis:")
        print(f"In-sample vs Out-sample difference: {diff_percent:.2e}%")

# plot that combines the fold comparison evolution and the cross validation results in two subplots in one figure
# use plot_fold_evolution() and plot_cross_validation() to get the data
 
    
def plot_combined_results(results, in_sample_means, out_sample_means,
                          in_sample_stds, out_sample_stds, strategies,
                          one_price_in, one_price_out, two_price_in, two_price_out, folds):
    """Plot the combined results of cross-validation and fold evolution"""
    
    # Create figure with two subplots stacked vertically (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18))  # Taller figure for vertical layout
    
    # --- Top subplot: Cross-validation results (bar chart) ---
    x = np.arange(len(strategies))
    width = 0.35
    
    # Create bars with custom colors and better styling
    bars1 = ax1.bar(x - width/2, in_sample_means, width, label='In-sample', 
           color='#2E86C1', yerr=in_sample_stds, capsize=5, 
           alpha=0.8, error_kw={'ecolor': '0.2', 'capthick': 2})
    bars2 = ax1.bar(x + width/2, out_sample_means, width, label='Out-of-sample', 
           color='#E67E22', yerr=out_sample_stds, capsize=5, 
           alpha=0.8, error_kw={'ecolor': '0.2', 'capthick': 2})
    
    # Add text annotations showing profit values in the middle of each bar
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{int(height):,}',
                ha='center', va='center', color='white', fontsize=18, fontweight='bold')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{int(height):,}',
                ha='center', va='center', color='white', fontsize=18, fontweight='bold')
    
    # Set y-axis limits to start from 300,000
    ax1.set_ylim(bottom=300000)
    
    ax1.set_xlabel('Strategy', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Expected Profit (EUR)', fontsize=22, fontweight='bold')
    ax1.set_title('Cross-validation Results', fontsize=22, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend(fontsize=22)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    
    # Format y-axis with comma separator
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # --- Bottom subplot: Fold evolution (line chart) ---
    # Plot fold-by-fold comparison with enhanced styling
    ax2.plot(folds, one_price_in, 'o-', color='#2E86C1', label='One-Price In-sample', 
            alpha=0.9, linewidth=2.5, markersize=8)
    ax2.plot(folds, one_price_out, 's-', color='#2E86C1', label='One-Price Out-of-sample', 
            linestyle='--', alpha=0.9, linewidth=2.5, markersize=8)
    ax2.plot(folds, two_price_in, 'o-', color='#E67E22', label='Two-Price In-sample', 
            alpha=0.9, linewidth=2.5, markersize=8)
    ax2.plot(folds, two_price_out, 's-', color='#E67E22', label='Two-Price Out-of-sample',
            linestyle='--', alpha=0.9, linewidth=2.5, markersize=8)
    
    # Add mean lines
    one_price_in_mean = np.mean(one_price_in)
    one_price_out_mean = np.mean(one_price_out)
    two_price_in_mean = np.mean(two_price_in)
    two_price_out_mean = np.mean(two_price_out)
    
    ax2.axhline(y=one_price_in_mean, color='#2E86C1', linestyle='-', alpha=0.3)
    ax2.axhline(y=one_price_out_mean, color='#2E86C1', linestyle='--', alpha=0.3)
    ax2.axhline(y=two_price_in_mean, color='#E67E22', linestyle='-', alpha=0.3)
    ax2.axhline(y=two_price_out_mean, color='#E67E22', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax2.set_xlabel('Fold Number', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Expected Profit (EUR)', fontsize=22, fontweight='bold')
    ax2.set_title('Profit Evolution Across Folds', fontsize=22, fontweight='bold')
    
    # Enhanced grid
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Move legend inside the plot
    ax2.legend(fontsize=22, ncol=1, loc='lower right')
    
    # Customize ticks
    ax2.set_xticks(folds)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    
    # Format y-axis with comma separator
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add overall title to the figure
    fig.suptitle('Cross-Validation Analysis of Pricing Strategies', 
                fontsize=22, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout()
    fig.subplots_adjust(top=0.95, hspace=0.3)  # Adjust top spacing for title and increase space between subplots
    plt.savefig('part1/results/step3/figures/combined_results.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print('\nPlotted combined results and saved to part1/results/step3/figures/combined_results.png')
    plt.close()