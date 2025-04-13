import numpy as np
import pandas as pd
from .step1_one_price import solve_one_price_offering_strategy
from .step2_two_price import solve_two_price_offering_strategy

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

def perform_cross_validation(all_scenarios, n_folds=8, in_sample_size=200, capacity_wind_farm=500, n_hours=24):
    """Perform k-fold cross-validation analysis"""
    # Convert scenarios dict to list for easier splitting
    scenario_keys = list(all_scenarios.keys())
    np.random.shuffle(scenario_keys)
    
    results = {
        'one_price': {'in_sample': [], 'out_sample': []},
        'two_price': {'in_sample': [], 'out_sample': []}
    }
    
    fold_size = len(scenario_keys) // n_folds
    
    for fold in range(n_folds):
        print(f"Processing fold {fold + 1}/{n_folds}")
        
        # Select scenarios for this fold
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_keys = scenario_keys[test_start:test_end]
        train_keys = [k for k in scenario_keys if k not in test_keys]
        
        # Randomly select in-sample scenarios from training set
        in_sample_keys = np.random.choice(train_keys, size=in_sample_size, replace=False)
        out_sample_keys = [k for k in test_keys]
        
        # Create scenario subsets
        in_sample_scenarios = {k: all_scenarios[k] for k in in_sample_keys}
        out_sample_scenarios = {k: all_scenarios[k] for k in out_sample_keys}
        
        # Solve strategies and calculate profits
        for strategy, solver in [
            ('one_price', solve_one_price_offering_strategy),
            ('two_price', solve_two_price_offering_strategy)
        ]:
            # Solve offering strategy
            offers, _, _ = solver(in_sample_scenarios, capacity_wind_farm, n_hours)
            
            # Calculate profits
            in_sample_profit = calculate_profits(
                offers, in_sample_scenarios, capacity_wind_farm, n_hours, strategy.split('_')[0]
            )
            out_sample_profit = calculate_profits(
                offers, out_sample_scenarios, capacity_wind_farm, n_hours, strategy.split('_')[0]
            )
            
            # Store results
            results[strategy]['in_sample'].append(in_sample_profit)
            results[strategy]['out_sample'].append(out_sample_profit)
    
    # Calculate summary statistics
    for strategy in results:
        for sample_type in ['in_sample', 'out_sample']:
            profits = results[strategy][sample_type]
            results[strategy][f'{sample_type}_avg'] = np.mean(profits)
            results[strategy][f'{sample_type}_std'] = np.std(profits)
    
    return results