import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_two_price_offering_strategy(in_sample_scenarios, capacity_wind_farm, n_hours):
    """
    Solve the two-price offering strategy by solving each hour independently.
    """
    optimal_offers = []
    total_expected_profit = 0
    scenario_profits = {s: 0 for s in in_sample_scenarios}

    
    # Solve for each hour separately
    for hour in range(n_hours):
        # Create optimization model for this hour
        model = gp.Model(f"WindFarmTwoPrice_Hour_{hour}")
        model.setParam('OutputFlag', 0)  # Suppress output
        
        # Decision variable: power to offer in day-ahead market for this hour
        p_DA = model.addVar(lb=0, ub=capacity_wind_farm, name=f"p_DA_{hour}")
        
        # Variables for positive and negative imbalances and profit calculation
        pos_imbalance = model.addVars(in_sample_scenarios.keys(), lb=0, name=f"pos_imbalance_{hour}")
        neg_imbalance = model.addVars(in_sample_scenarios.keys(), lb=0, name=f"neg_imbalance_{hour}")
        profit = model.addVars(in_sample_scenarios.keys(), name=f"profit_{hour}")
        
        # Calculate imbalance and profit for each scenario
        for s in in_sample_scenarios:
            scenario = in_sample_scenarios[s]
            
            # Extract data for this hour and scenario
            wind_actual = scenario.loc[hour, 'wind']
            price_DA = scenario.loc[hour, 'price']
            price_BAL = scenario.loc[hour, 'balancing_price']
            condition = scenario.loc[hour, 'condition']  # 0=excess, 1=deficit
            
            # Imbalance constraints
            model.addConstr(wind_actual - p_DA == pos_imbalance[s] - neg_imbalance[s], f"imbalance_{s}")
            
            # Two-price balancing scheme
            if condition == 0:  # System excess
                # Surplus at balancing price, deficit at day-ahead
                surplus_price = price_BAL # balance price is lower
                deficit_price = price_DA # day-ahead price is higher
            else:  # System deficit (condition == 1)
                # Surplus at day-ahead, deficit at balancing price
                surplus_price = price_DA # day ahead price is lower
                deficit_price = price_BAL # balance price is higher
            
            # Profit calculation for this scenario and hour
            profit_expr = price_DA * p_DA + surplus_price * pos_imbalance[s] - deficit_price * neg_imbalance[s]
            model.addConstr(profit[s] == profit_expr, f"profit_scenario_{s}")
        
        # Objective: maximize expected profit for this hour
        expected_profit = gp.LinExpr()
        n_scenarios = len(in_sample_scenarios)
        PROBABILITY = 1.0 / n_scenarios
        for s in in_sample_scenarios:
            expected_profit.addTerms(PROBABILITY, profit[s])
        
        model.setObjective(expected_profit, GRB.MAXIMIZE)
        
        # Solve model for this hour
        try:
            model.optimize()
            
            # Check if the model was solved successfully
            if model.status == GRB.OPTIMAL:
                # Save results for this hour
                optimal_offers.append(p_DA.X)
                hour_profit = model.objVal
                total_expected_profit += hour_profit
                
                # Add each scenario's profit for this hour to total
                for s in in_sample_scenarios:
                    scenario_profits[s] += profit[s].X
            else:
                print(f"Warning: Hour {hour} optimization status: {model.status}")
                # Use expected wind production as fallback strategy
                # Use the average wind production across scenarios for this hour
                avg_wind = sum(in_sample_scenarios[s].loc[hour, 'wind'] for s in in_sample_scenarios) / len(in_sample_scenarios)
                # avg_wind = in_sample_scenarios[s].loc[hour, 'wind']
                optimal_offers.append(min(avg_wind, capacity_wind_farm))
                # We can't get profits from an unsolved model, so we'll use an estimate
                for s in in_sample_scenarios:
                    scenario = in_sample_scenarios[s]
                    wind_actual = scenario.loc[hour, 'wind']
                    price_DA = scenario.loc[hour, 'price']
                    price_BAL = scenario.loc[hour, 'balancing_price']
                    condition = scenario.loc[hour, 'condition']
                    
                    # Estimate profit using our fallback strategy
                    if condition == 0:  # System excess
                        surplus_price = price_BAL
                        deficit_price = price_DA
                    else:  # System deficit
                        surplus_price = price_DA
                        deficit_price = price_BAL
                        
                    imbalance = wind_actual - optimal_offers[-1]
                    hour_profit = price_DA * optimal_offers[-1]
                    if imbalance > 0:  # Surplus
                        hour_profit += surplus_price * imbalance
                    else:  # Deficit
                        hour_profit += deficit_price * imbalance
                    
                    scenario_profits[s] += hour_profit
                    
        except Exception as e:
            print(f"Error solving hour {hour}: {e}")
            # Use expected wind production as fallback
            avg_wind = sum(in_sample_scenarios[s].loc[hour, 'wind'] for s in in_sample_scenarios) / len(in_sample_scenarios)
            optimal_offers.append(min(avg_wind, capacity_wind_farm))
    
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