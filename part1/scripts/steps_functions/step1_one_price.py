# Standard library imports
import os
import pickle

# Third-party imports
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def solve_one_price_offering_strategy(in_sample_scenarios, CAPACITY_WIND_FARM, N_HOURS):
    """
    Solve the stochastic offering strategy problem under a one-price balancing scheme.
    
    In a one-price balancing scheme, both positive and negative imbalances are settled
    at the same balancing price. This model determines the optimal hourly day-ahead market
    offers that maximize expected profit across all scenarios.
    
    Args:
        in_sample_scenarios: Dictionary of scenarios keyed by scenario IDs, each containing
                           hourly wind production and prices
        CAPACITY_WIND_FARM: Maximum capacity of the wind farm in MW
        N_HOURS: Number of hours in the planning horizon (typically 24)
    
    Returns:
        optimal_offers_one_price: List of optimal hourly production quantity offers (MW)
        expected_profit: Expected profit across all scenarios
        scenario_profits: Dictionary mapping scenario IDs to their profit values
    """
    # --- Setup constants and model parameters ---
    N_SCENARIOS = len(in_sample_scenarios)
    SCENARIO_PROB = 1.0 / N_SCENARIOS  # Equal probability for each scenario (uniform distribution)
    
    # --- Create optimization model ---
    model = gp.Model("WindFarmOnePrice")
    model.setParam('OutputFlag', 0)  # Optional: suppress Gurobi output
    
    # --- Decision variables ---
    # p_DA[h]: Power offered in day-ahead market for hour h (MW)
    # These are the primary decision variables we're solving for
    p_DA = model.addVars(N_HOURS, lb=0, ub=CAPACITY_WIND_FARM, name="p_DA")
    
    # profit[s]: Total profit for scenario s (€)
    # These variables track the profit calculation for each scenario
    profit = model.addVars(in_sample_scenarios.keys(), name="profit")
    
    # --- Calculate profit for each scenario ---
    for s in in_sample_scenarios:  # Iterate through each scenario
        scenario = in_sample_scenarios[s]  # Get data for this specific scenario
        profit_expr = gp.LinExpr(0)  # Initialize profit expression at zero
        
        # Calculate profit contribution from each hour
        for h in range(N_HOURS):
            # Extract scenario data for this hour
            wind_actual = scenario.loc[h, 'wind']  # Actual wind production (MW)
            price_DA = scenario.loc[h, 'price']    # Day-ahead market price (€/MW)
            price_BAL = scenario.loc[h, 'balancing_price']  # Balancing market price (€/MW)
            
            # --- Profit calculation components ---
            # Component 1: Revenue from day-ahead market
            # Day-ahead revenue = price_DA * p_DA[h]
            profit_expr.addTerms(price_DA, p_DA[h])
            
            # Component 2: Settlement in balancing market
            # In one-price scheme, imbalance is settled as: price_BAL * (wind_actual - p_DA[h])
            # This is separated into:
            #   - Constant term: price_BAL * wind_actual (not dependent on decision variable)
            profit_expr.addConstant(price_BAL * wind_actual)
            #   - Variable term: -price_BAL * p_DA[h] (dependent on decision variable)
            profit_expr.addTerms(-price_BAL, p_DA[h])
            
            # Combined profit for hour h:
            # profit[h] = price_DA * p_DA[h] + price_BAL * wind_actual - price_BAL * p_DA[h]
            #           = p_DA[h] * (price_DA - price_BAL) + price_BAL * wind_actual
            #
            # Economic interpretation:
            # - If price_DA > price_BAL: Producer benefits from offering more in day-ahead market
            # - If price_DA < price_BAL: Producer benefits from offering less in day-ahead market
        
        # Add constraint defining total profit for this scenario
        model.addConstr(profit[s] == profit_expr, f"profit_scenario_{s}")
    
    # --- Objective function: Maximize expected profit across all scenarios ---
    # For stochastic optimization, we take the probability-weighted sum of all scenario profits
    expected_profit = gp.LinExpr()
    for s in in_sample_scenarios:
        expected_profit.addTerms(SCENARIO_PROB, profit[s])  # Add weighted profit from each scenario
    
    model.setObjective(expected_profit, GRB.MAXIMIZE)
    
    # --- Solve the optimization model ---
    model.optimize()
    
    # --- Extract and return results ---
    # Convert optimal variable values to Python list/dict for return
    optimal_offers_one_price = [p_DA[h].X for h in range(N_HOURS)]  # Optimal hourly offers
    scenario_profits = {s: profit[s].X for s in in_sample_scenarios}  # Profit for each scenario
    
    return optimal_offers_one_price, model.objVal, scenario_profits