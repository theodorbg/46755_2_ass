import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_two_price_offering_strategy(in_sample_scenarios, capacity_wind_farm,
                                      n_hours):
    """
    Solve the two-price offering strategy problem.
    
    Args:
        in_sample_scenarios: Dictionary of in-sample scenarios
        capacity_wind_farm: Maximum capacity of the wind farm (MW)
        n_hours: Number of hours in planning horizon
        
    Returns:
        Optimal day-ahead offers for the wind farm
    """
    # Initialize parameters
    n_scenarios = len(in_sample_scenarios)
    probability = 1.0 / n_scenarios

    # Create price arrays for each scenario and hour
    surplus_price = np.zeros((n_hours, n_scenarios))
    deficit_price = np.zeros((n_hours, n_scenarios))
    
    # Set prices based on system conditions
    for s in range(1, n_scenarios + 1):
        condition = in_sample_scenarios[s]['condition']  # one day
        for t in range(n_hours):
            # Two-price balancing scheme
            if condition[t] == 0:  # System excess
                # Surplus at balancing price, deficit at day-ahead
                surplus_price[t, s-1] = in_sample_scenarios[s]['balancing_price'].iloc[t]
                deficit_price[t, s-1] = in_sample_scenarios[s]['price'].iloc[t]
            else:  # System deficit (condition == 1)
                # Surplus at day-ahead, deficit at balancing price
                surplus_price[t, s-1] = in_sample_scenarios[s]['price'].iloc[t]
                deficit_price[t, s-1] = in_sample_scenarios[s]['balancing_price'].iloc[t]

    # Create optimization model
    model = gp.Model("WindFarmTwoPrice_Hour")
    # model.setParam('OutputFlag', 0)  # Suppress output
        
    # Decision variable: day-ahead market offers
    p_da = model.addMVar(
        shape=(n_hours), 
        lb=0, 
        ub=capacity_wind_farm, 
        name="p_DA"
    )
    
    # Variables for positive and negative imbalances
    pos_imbalance = model.addMVar(
        shape=(n_hours, n_scenarios), 
        lb=0, 
        name="pos_imbalance"
    )
    neg_imbalance = model.addMVar(
        shape=(n_hours, n_scenarios), 
        lb=0, 
        name="neg_imbalance"
    )
    
    # Objective function - maximize expected profit
    objective_expr = gp.quicksum(
        gp.quicksum(
            probability * (
                in_sample_scenarios[s]['price'].iloc[t] * p_da[t] +
                surplus_price[t, s-1] * pos_imbalance[t, s-1] -
                deficit_price[t, s-1] * neg_imbalance[t, s-1]
            )
            for s in range(1, n_scenarios + 1)
        )
        for t in range(n_hours)
    )
    model.setObjective(objective_expr, GRB.MAXIMIZE)
    
    # Imbalance constraints
    for t in range(n_hours):
        for s in range(1, n_scenarios + 1):
            wind_actual = in_sample_scenarios[s]['wind'].iloc[t]
            model.addConstr(
                wind_actual - p_da[t] == 
                pos_imbalance[t, s-1] - neg_imbalance[t, s-1],
                f"imbalance_{t}_{s}"
            )
    
    # Solve model
    model.optimize()
    model.write("two_price_model.lp")  # Save model for debugging
    
    print(model.status)
    
    # Check if the model was solved successfully
    if model.status == GRB.OPTIMAL:
        # Extract results
        optimal_offers = [p_da[t].X for t in range(n_hours)]
        total_profit = model.objVal
        return optimal_offers, total_profit
    else:
        # Handle non-optimal status
        print(f"Warning: Optimization status: {model.status}")
        return None


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