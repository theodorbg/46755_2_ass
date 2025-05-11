import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt


import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Define constants if they are not globally available or passed as arguments
EXCESS_FACTOR = 0.85
DEFICIT_FACTOR = 1.25

def solve_two_price_offering_strategy(in_sample_scenarios, capacity_wind_farm, n_hours):
    """
    Solve the two-price offering strategy problem.
    
    Args:
        in_sample_scenarios: Dictionary of in-sample scenarios (keys can be non-sequential)
        capacity_wind_farm: Maximum capacity of the wind farm (MW)
        n_hours: Number of hours in planning horizon
        
    Returns:
        optimal_offers: List of optimal day-ahead offers for each hour.
        total_profit: Total expected profit over all scenarios.
        scenario_profits: Dictionary with profit for each scenario, keyed by original scenario keys.
    """
    n_scenarios = len(in_sample_scenarios)
    if n_scenarios == 0:
        # Handle empty scenarios case
        return [0] * n_hours, 0, {}
        
    probability = 1.0 / n_scenarios
    
    # Create a mapping from original scenario keys to 0-based indices
    s_keys = list(in_sample_scenarios.keys())
    scenario_idx_map = {s_key: i for i, s_key in enumerate(s_keys)}
    
    # Initialize dictionary to track profits per scenario, using original keys
    calculated_scenario_profits = {s_key: 0 for s_key in s_keys}

    model = gp.Model("WindFarmTwoPrice")
    #model.setParam('DualReductions', 0)
    #model.setParam('OutputFlag', 0) # Suppress Gurobi output for cleaner logs
        
    p_da = model.addMVar(
        shape=(n_hours), 
        lb=0, 
        ub=capacity_wind_farm,
        name="p_DA"
    )
    
    pos_imbalance = model.addMVar(
        shape=(n_hours, n_scenarios), # Indexed 0 to n_scenarios-1
        lb=0,
        name="pos_imbalance"
    )
    neg_imbalance = model.addMVar(
        shape=(n_hours, n_scenarios), # Indexed 0 to n_scenarios-1
        lb=0,
        name="neg_imbalance"
    )
    
    objective_expr = gp.quicksum(
        probability * (
            in_sample_scenarios[s_key]['price'].iloc[t] * p_da[t] 
            + in_sample_scenarios[s_key]['condition'].iloc[t] * ( # System DEFICIT (condition=1)
                pos_imbalance[t, scenario_idx_map[s_key]] * in_sample_scenarios[s_key]['price'].iloc[t] # Producer surplus, paid at DA price
                - neg_imbalance[t, scenario_idx_map[s_key]] * in_sample_scenarios[s_key]['price'].iloc[t] * DEFICIT_FACTOR # Producer deficit, pays higher price
            )
            + (1 - in_sample_scenarios[s_key]['condition'].iloc[t]) * ( # System EXCESS (condition=0)
                pos_imbalance[t, scenario_idx_map[s_key]] * in_sample_scenarios[s_key]['price'].iloc[t] * EXCESS_FACTOR # Producer surplus, paid lower price
                - neg_imbalance[t, scenario_idx_map[s_key]] * in_sample_scenarios[s_key]['price'].iloc[t] # Producer deficit, pays DA price
            )
        )
        for s_key in s_keys # Iterate over original keys
        for t in range(n_hours)
    )
    model.setObjective(objective_expr, GRB.MAXIMIZE)
    
    for t in range(n_hours):
        for s_key in s_keys: # Iterate over original keys
            wind_actual = in_sample_scenarios[s_key]['wind'].iloc[t]
            idx = scenario_idx_map[s_key] # Get 0-based index
            model.addConstr(
                wind_actual - p_da[t] == 
                pos_imbalance[t, idx] - neg_imbalance[t, idx],
                f"imbalance_{t}_{s_key}" # Use original key for unique constraint name
            )
            
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        optimal_offers = [p_da[t].X for t in range(n_hours)]
        total_profit = model.objVal
        
        for s_key in s_keys: # Iterate over original keys
            current_scenario_profit = 0
            idx = scenario_idx_map[s_key] # Get 0-based index
            for t in range(n_hours):
                day_ahead_revenue = in_sample_scenarios[s_key]['price'].iloc[t] * optimal_offers[t]
                
                pos_imb_val = pos_imbalance[t, idx].X
                neg_imb_val = neg_imbalance[t, idx].X
                
                condition = in_sample_scenarios[s_key]['condition'].iloc[t]
                balancing_revenue = 0
                if condition == 1:  # System DEFICIT
                    balancing_revenue = (pos_imb_val * in_sample_scenarios[s_key]['price'].iloc[t] -
                                         neg_imb_val * in_sample_scenarios[s_key]['price'].iloc[t] * DEFICIT_FACTOR)
                else:  # System EXCESS
                    balancing_revenue = (pos_imb_val * in_sample_scenarios[s_key]['price'].iloc[t] * EXCESS_FACTOR -
                                         neg_imb_val * in_sample_scenarios[s_key]['price'].iloc[t])
                
                current_scenario_profit += day_ahead_revenue + balancing_revenue
            calculated_scenario_profits[s_key] = current_scenario_profit
        
        return optimal_offers, total_profit, calculated_scenario_profits
    else:
        # print(f"Warning: Two-price optimization status for a fold: {model.status}") # For debugging
        # Return dummy/default values ensuring three values are returned
        return [0.0] * n_hours, 0.0, {s_key: 0.0 for s_key in s_keys}

def solve_two_price_offering_strategy_hourly(in_sample_scenarios, capacity_wind_farm, n_hours):
    """
    Solve the two-price offering strategy by solving each hour independently.
    """
    optimal_offers = []
    total_expected_profit = 0
    scenario_profits = {s: 0 for s in in_sample_scenarios}
    
    # Solve for each hour separately
    for hour in range(n_hours):
        # Create hour-specific model
        model = gp.Model(f"WindFarmTwoPrice_Hour_{hour}")
        model.setParam('OutputFlag', 0)  # Suppress solver output
        
        # Single decision variable for this hour's day-ahead offer
        p_DA = model.addVar(lb=0, ub=capacity_wind_farm, name=f"p_DA_{hour}")
        
        # Variables for each scenario's imbalances
        pos_imbalance = model.addVars(in_sample_scenarios.keys(), lb=0, name=f"pos_imbalance_{hour}")
        neg_imbalance = model.addVars(in_sample_scenarios.keys(), lb=0, name=f"neg_imbalance_{hour}")
        profit = model.addVars(in_sample_scenarios.keys(), name=f"profit_{hour}")
        
        # Set up constraints for each scenario
        for s in in_sample_scenarios:
            # Extract scenario data for this hour
            wind_actual = in_sample_scenarios[s]['wind'].iloc[hour]
            price_DA = in_sample_scenarios[s]['price'].iloc[hour]
            price_BAL = in_sample_scenarios[s]['balancing_price'].iloc[hour]
            condition = in_sample_scenarios[s]['condition'].iloc[hour]  # 0=excess, 1=deficit
            
            # Imbalance constraint
            model.addConstr(wind_actual - p_DA == pos_imbalance[s] - neg_imbalance[s], f"imbalance_{s}")
            
            # Price determination based on system condition
            if condition == 0:  # System excess
                surplus_price = price_BAL
                deficit_price = price_DA
            else:  # System deficit
                surplus_price = price_DA
                deficit_price = price_BAL
            
            # Profit calculation
            profit_expr = price_DA * p_DA + surplus_price * pos_imbalance[s] - deficit_price * neg_imbalance[s]
            model.addConstr(profit[s] == profit_expr, f"profit_calc_{s}")
        
        # Objective: maximize expected profit across all scenarios
        n_scenarios = len(in_sample_scenarios)
        probability = 1.0 / n_scenarios
        
        model.setObjective(
            gp.quicksum(probability * profit[s] for s in in_sample_scenarios), 
            GRB.MAXIMIZE
        )
        
        # Solve model for this hour
        model.optimize()
        
        # Store results
        if model.status == GRB.OPTIMAL:
            optimal_offers.append(p_DA.X)
            
            # Record profits for each scenario
            for s in in_sample_scenarios:
                scenario_profits[s] += profit[s].X
                
            # Add to total expected profit
            total_expected_profit += model.objVal
        else:
            print(f"Warning: Hour {hour} optimization status: {model.status}")

            # Make fallback strategy: just bid half the capacity of the wind farm
            optimal_offers.append(capacity_wind_farm / 2)
            # Raise exception instead of using fallback
            # raise Exception(f"Model for hour {hour} could not be solved optimally (status {model.status})")
    
    return optimal_offers, total_expected_profit, scenario_profits


def forecast_strategy(in_sample_scenarios, capacity_wind_farm, n_hours):
    """
    Solve using a simple expected wind production strategy (no optimization).
    Bid the average expected wind production for each hour.
    
    Args:
        in_sample_scenarios: Dictionary of in-sample scenarios
        capacity_wind_farm: Maximum capacity of the wind farm (MW)
        n_hours: Number of hours in planning horizon
        
    Returns:
        optimal_offers: Hourly production quantity offers based on expected wind
        expected_profit: Expected profit
        scenario_profits: Profits for each scenario
    """
    optimal_offers = []
    total_expected_profit = 0
    scenario_profits = {s: 0 for s in in_sample_scenarios}
    
    # For each hour, use the average wind production as the bid
    for hour in range(n_hours):
        # Calculate average wind production for this hour across all scenarios
        avg_wind = sum(in_sample_scenarios[s].loc[hour, 'wind'] for s in in_sample_scenarios) / len(in_sample_scenarios)
        
        # Cap the offer at the wind farm capacity
        optimal_offers.append(min(avg_wind, capacity_wind_farm))
        
        # Calculate profit for each scenario with this bidding strategy
        hour_profit_total = 0
        
        for s in in_sample_scenarios:
            scenario = in_sample_scenarios[s]
            
            # Extract data for this hour and scenario
            wind_actual = scenario.loc[hour, 'wind']
            price_DA = scenario.loc[hour, 'price']
            price_BAL = scenario.loc[hour, 'balancing_price']
            condition = scenario.loc[hour, 'condition']  # 0=excess, 1=deficit
            
            # Two-price balancing scheme pricing
            if condition == 0:  # System excess
                surplus_price = price_BAL  # Lower in excess
                deficit_price = price_DA   # Regular price for deficit
            else:  # System deficit (condition == 1)
                surplus_price = price_DA   # Regular price for surplus
                deficit_price = price_BAL  # Higher in deficit
            
            # Calculate imbalance
            imbalance = wind_actual - optimal_offers[-1]
            
            # Calculate profit
            hour_profit = price_DA * optimal_offers[-1]  # Day-ahead market revenue
            
            if imbalance > 0:  # Surplus: produced more than offered
                hour_profit += surplus_price * imbalance
            else:  # Deficit: produced less than offered
                hour_profit += deficit_price * imbalance  # Note: imbalance is negative, so this is a reduction
            
            # Add to scenario profits
            scenario_profits[s] += hour_profit
            hour_profit_total += hour_profit
        
        # Calculate expected profit for this hour
        hour_expected_profit = hour_profit_total / len(in_sample_scenarios)
        total_expected_profit += hour_expected_profit
    
    return optimal_offers, total_expected_profit, scenario_profits

def compare_one_price_two_price(expected_profit_one_price, two_price_total_expected_profit):
    """ Compare the expected profits of one-price and two-price strategies. """
    print("\n=== COMPARISON: ONE-PRICE vs TWO-PRICE ===")
    print(f"One-Price Expected Profit: {expected_profit_one_price:.2e} EUR")
    print(f"Two-Price Expected Profit: {two_price_total_expected_profit:.2e} EUR")
    print(f"Difference: {two_price_total_expected_profit - expected_profit_one_price:.2e} EUR")

    return None

def compare_all_strategies(expected_profit_one_price,
                           two_price_total_expected_profit,
                           ew_expected_profit):
    print("\n=== COMPARISON: ALL STRATEGIES ===")

    print(f"One-Price Expected Profit: "
          f"{expected_profit_one_price:.2e} EUR")
    
    print(f"Two-Price Expected Profit: "
          f"{two_price_total_expected_profit:.2e} EUR")
    
    print(f"Expected Wind Profit: "
          f"{ew_expected_profit:.2e} EUR")
    
    print(f"Expected Wind vs One-Price: "
          f"{ew_expected_profit - expected_profit_one_price:.2e} EUR")
    
    print(f"Expected Wind vs Two-Price: "
          f"{ew_expected_profit - two_price_total_expected_profit:.2e} EUR")

    return None
