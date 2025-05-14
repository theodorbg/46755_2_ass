# Standard imports for mathematical optimization, array handling and visualization
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# Constants for two-price balancing scheme
EXCESS_FACTOR = 0.85  # Factor applied to day-ahead price during system excess (producer surplus gets 85% of day-ahead price)
DEFICIT_FACTOR = 1.25  # Factor applied to day-ahead price during system deficit (producer deficit pays 125% of day-ahead price)

def solve_two_price_offering_strategy(in_sample_scenarios, capacity_wind_farm, n_hours):
    """
    Solve the two-price offering strategy problem.
    
    In a two-price balancing scheme, positive and negative imbalances are settled at different prices
    depending on the overall system state (excess or deficit). This model determines the optimal
    hourly day-ahead market offers that maximize expected profit across all scenarios considering
    these dual-pricing rules.
    
    Args:
        in_sample_scenarios: Dictionary of in-sample scenarios (keys can be non-sequential)
        capacity_wind_farm: Maximum capacity of the wind farm (MW)
        n_hours: Number of hours in planning horizon
        
    Returns:
        optimal_offers: List of optimal day-ahead offers for each hour.
        total_profit: Total expected profit over all scenarios.
        scenario_profits: Dictionary with profit for each scenario, keyed by original scenario keys.
    """
    # --- Initialize model parameters ---
    n_scenarios = len(in_sample_scenarios)
    if n_scenarios == 0:
        # Edge case: no scenarios provided, return zeros
        return [0] * n_hours, 0, {}
        
    probability = 1.0 / n_scenarios  # Equal probability for each scenario
    
    # --- Handle non-sequential scenario keys ---
    # Map original scenario keys to sequential indices for use with MVar
    # This is needed because MVar requires zero-based sequential indices
    s_keys = list(in_sample_scenarios.keys())
    scenario_idx_map = {s_key: i for i, s_key in enumerate(s_keys)}
    
    # Storage for calculated profits with original scenario keys
    calculated_scenario_profits = {s_key: 0 for s_key in s_keys}

    # --- Create optimization model ---
    model = gp.Model("WindFarmTwoPrice")
    #model.setParam('DualReductions', 0)  # Optional: may help solve edge cases
    #model.setParam('OutputFlag', 0)      # Optional: suppress Gurobi output
        
    # --- Decision variables ---
    # p_da: Power offered in day-ahead market for each hour (MW)
    p_da = model.addMVar(
        shape=(n_hours), 
        lb=0, 
        ub=capacity_wind_farm,  # Cannot exceed wind farm capacity
        name="p_DA"
    )
    
    # pos_imbalance: Positive imbalance (when actual production > day-ahead offer) for each hour and scenario
    pos_imbalance = model.addMVar(
        shape=(n_hours, n_scenarios),  # Indexed by hour and scenario
        lb=0,                         # Must be non-negative
        name="pos_imbalance"
    )
    
    # neg_imbalance: Negative imbalance (when actual production < day-ahead offer) for each hour and scenario
    neg_imbalance = model.addMVar(
        shape=(n_hours, n_scenarios),  # Indexed by hour and scenario
        lb=0,                         # Must be non-negative
        name="neg_imbalance"
    )
    
    # --- Objective function: Maximize expected profit ---
    # The objective accounts for:
    # 1. Day-ahead market revenue (price_DA * p_da)
    # 2. Balancing market settlements, which depends on:
    #    - Whether we have surplus (pos_imbalance) or deficit (neg_imbalance)
    #    - System condition (excess or deficit)
    objective_expr = gp.quicksum(
        probability * (
            # Part 1: Day-ahead market revenue
            in_sample_scenarios[s_key]['price'].iloc[t] * p_da[t] 
            
            # Part 2A: Balancing under system DEFICIT condition
            + in_sample_scenarios[s_key]['condition'].iloc[t] * (
                # When producer has surplus (pos_imbalance):
                # Producer is paid day-ahead price (favorable to producer)
                pos_imbalance[t, scenario_idx_map[s_key]] * in_sample_scenarios[s_key]['price'].iloc[t]
                
                # When producer has deficit (neg_imbalance):
                # Producer pays penalty price (unfavorable to producer)
                - neg_imbalance[t, scenario_idx_map[s_key]] * in_sample_scenarios[s_key]['price'].iloc[t] * DEFICIT_FACTOR
            )
            
            # Part 2B: Balancing under system EXCESS condition
            + (1 - in_sample_scenarios[s_key]['condition'].iloc[t]) * (
                # When producer has surplus (pos_imbalance):
                # Producer is paid reduced price (unfavorable to producer)
                pos_imbalance[t, scenario_idx_map[s_key]] * in_sample_scenarios[s_key]['price'].iloc[t] * EXCESS_FACTOR
                
                # When producer has deficit (neg_imbalance):
                # Producer pays day-ahead price (favorable to producer)
                - neg_imbalance[t, scenario_idx_map[s_key]] * in_sample_scenarios[s_key]['price'].iloc[t]
            )
        )
        for s_key in s_keys  # Iterate over all scenarios
        for t in range(n_hours)  # Iterate over all hours
    )
    model.setObjective(objective_expr, GRB.MAXIMIZE)
    
    # --- Imbalance constraints ---
    # For each hour and scenario, the imbalance (difference between actual wind production
    # and day-ahead offer) must equal pos_imbalance - neg_imbalance
    # Only one of pos_imbalance or neg_imbalance will be non-zero due to cost minimization
    for t in range(n_hours):
        for s_key in s_keys:
            wind_actual = in_sample_scenarios[s_key]['wind'].iloc[t]  # Actual wind production
            idx = scenario_idx_map[s_key]  # Get mapped index for MVars
            
            model.addConstr(
                wind_actual - p_da[t] == pos_imbalance[t, idx] - neg_imbalance[t, idx],
                f"imbalance_{t}_{s_key}"  # Unique constraint name
            )
            
    # --- Solve the model ---
    model.optimize()
    
    # --- Process results ---
    if model.status == GRB.OPTIMAL:
        # Extract optimal day-ahead offers
        optimal_offers = [p_da[t].X for t in range(n_hours)]
        total_profit = model.objVal  # Total expected profit
        
        # Calculate profit for each scenario under optimal strategy
        for s_key in s_keys:
            current_scenario_profit = 0
            idx = scenario_idx_map[s_key]
            
            for t in range(n_hours):
                # Day-ahead market revenue
                day_ahead_revenue = in_sample_scenarios[s_key]['price'].iloc[t] * optimal_offers[t]
                
                # Get imbalance values
                pos_imb_val = pos_imbalance[t, idx].X  # Positive imbalance (surplus)
                neg_imb_val = neg_imbalance[t, idx].X  # Negative imbalance (deficit)
                
                # Get system condition for this scenario and hour
                condition = in_sample_scenarios[s_key]['condition'].iloc[t]  # 0=excess, 1=deficit
                
                # Calculate balancing market revenue/cost
                balancing_revenue = 0
                if condition == 1:  # System DEFICIT
                    # Surplus paid at day-ahead price, deficit penalized
                    balancing_revenue = (pos_imb_val * in_sample_scenarios[s_key]['price'].iloc[t] -
                                         neg_imb_val * in_sample_scenarios[s_key]['price'].iloc[t] * DEFICIT_FACTOR)
                else:  # System EXCESS
                    # Surplus paid at reduced price, deficit at day-ahead price
                    balancing_revenue = (pos_imb_val * in_sample_scenarios[s_key]['price'].iloc[t] * EXCESS_FACTOR -
                                         neg_imb_val * in_sample_scenarios[s_key]['price'].iloc[t])
                
                # Total hour profit for this scenario
                current_scenario_profit += day_ahead_revenue + balancing_revenue
                
            # Store calculated profit
            calculated_scenario_profits[s_key] = current_scenario_profit
        
        return optimal_offers, total_profit, calculated_scenario_profits
    else:
        # Model was not solved optimally
        # Return default values rather than failing
        return [0.0] * n_hours, 0.0, {s_key: 0.0 for s_key in s_keys}

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
