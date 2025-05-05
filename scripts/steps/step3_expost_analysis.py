import numpy as np
import pandas as pd
from .step1_one_price import solve_one_price_offering_strategy
from .step2_two_price import solve_two_price_offering_strategy
from .step2_two_price import solve_two_price_offering_strategy_hourly

def calculate_profits(offers, scenarios, capacity_wind_farm, n_hours, price_scheme='one'):
    """Calculate profits for a set of scenarios given the offering decisions"""
    total_profit = 0
    
    for s in scenarios:
        scenario = scenarios[s]
        scenario_profit = 0
        
        for h in range(n_hours):
            wind_actual = scenario.loc[h, 'wind']
            price_DA = scenario.loc[h, 'price']
            price_BAL = scenario.loc[h, 'balancing_price']
            condition = scenario.loc[h, 'condition']
            
            # Day-ahead revenue
            p_DA = offers[h]
            revenue_DA = price_DA * p_DA
            
            # Balancing settlement
            imbalance = wind_actual - p_DA
            
            if price_scheme == 'one':
                revenue_BAL = price_BAL * imbalance
            else:
                if condition == 0:  # System excess
                    revenue_BAL = price_BAL * max(0, imbalance) + price_DA * min(0, imbalance)
                else:  # System deficit
                    revenue_BAL = price_DA * max(0, imbalance) + price_BAL * min(0, imbalance)
            
            scenario_profit += revenue_DA + revenue_BAL
        
        total_profit += scenario_profit
    
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
    
    # Calculate in_sample_size
    in_sample_size = len(in_sample_scenarios)
    
    # Create list of all scenario keys from both dictionaries
    all_keys = list(in_sample_scenarios.keys()) + list(out_sample_scenarios.keys())
    
    # Process each fold
    for fold in range(n_folds):
        print(f"Processing fold {fold + 1}/{n_folds}")
        
        if fold == 0:
            # First fold uses original in-sample scenarios (0-199)
            fold_in_sample = in_sample_scenarios
            fold_out_sample = out_sample_scenarios
        else:
            # For subsequent folds, select a different block of scenarios as in-sample
            # Calculate range of keys to use from out_sample_scenarios
            start_idx = (fold - 1) * in_sample_size
            end_idx = start_idx + in_sample_size
            
            if start_idx >= len(out_sample_scenarios):
                print(f"Warning: Not enough scenarios for fold {fold + 1}")
                break
                
            # Select keys for this fold's in-sample
            out_sample_keys = list(out_sample_scenarios.keys())
            fold_in_sample_keys = out_sample_keys[start_idx:min(end_idx, len(out_sample_keys))]
            
            # All other keys become out-of-sample
            fold_out_sample_keys = (list(in_sample_scenarios.keys()) + 
                                   [k for k in out_sample_scenarios.keys() if k not in fold_in_sample_keys])
            
            # Create scenario subsets for this fold
            fold_in_sample = {k: out_sample_scenarios[k] for k in fold_in_sample_keys}
            fold_out_sample = {}
            
            # Add original in-sample scenarios to out-sample
            for k in in_sample_scenarios:
                fold_out_sample[k] = in_sample_scenarios[k]
                
            # Add remaining out-sample scenarios to out-sample
            for k in out_sample_scenarios:
                if k not in fold_in_sample_keys:
                    fold_out_sample[k] = out_sample_scenarios[k]
        
        print(f"  In-sample size: {len(fold_in_sample)}, Out-sample size: {len(fold_out_sample)}")
        
        # Solve strategies and calculate profits
        for strategy, solver in [
            ('one_price', solve_one_price_offering_strategy),
            ('two_price', solve_two_price_offering_strategy)
        ]:
            # Use hourly solver for two-price to avoid license issues
            if solver.__name__ == 'solve_two_price_offering_strategy':
                solver = solve_two_price_offering_strategy_hourly
                
            # Solve offering strategy
            offers, _, _ = solver(fold_in_sample, capacity_wind_farm, n_hours)
            
            # Calculate profits
            in_sample_profit = calculate_profits(
                offers, fold_in_sample, capacity_wind_farm, n_hours, 
                strategy.split('_')[0]
            )
            out_sample_profit = calculate_profits(
                offers, fold_out_sample, capacity_wind_farm, n_hours,
                strategy.split('_')[0]
            )
            
            # Store results
            results[strategy]['in_sample'].append(in_sample_profit)
            results[strategy]['out_sample'].append(out_sample_profit)
    
    # Calculate summary statistics
    for strategy in results:
        for sample_type in ['in_sample', 'out_sample']:
            profits = results[strategy][sample_type]
            if profits:  # Check if list is not empty
                results[strategy][f'{sample_type}_avg'] = np.mean(profits)
                results[strategy][f'{sample_type}_std'] = np.std(profits)
            else:
                results[strategy][f'{sample_type}_avg'] = 0
                results[strategy][f'{sample_type}_std'] = 0
    
    return results