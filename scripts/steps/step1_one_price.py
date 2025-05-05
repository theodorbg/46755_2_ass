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
    
    Args:
        in_sample_scenarios: Dictionary of in-sample scenarios
    
    Returns:
        optimal_offers_one_price: Optimal hourly production quantity offers
        expected_profit: Expected profit
        scenario_profits: Profits for each scenario
    """
    # Constants
    # CAPACITY_WIND_FARM = 500  # MW
    N_SCENARIOS = len(in_sample_scenarios)
    SCENARIO_PROB = 1.0 / N_SCENARIOS  # Equal probability for each scenario
    
    # Create optimization model
    model = gp.Model("WindFarmOnePrice")
    
    # Decision variables: power to offer in day-ahead market for each hour
    p_DA = model.addVars(N_HOURS, lb=0, ub=CAPACITY_WIND_FARM, name="p_DA")
    
    # Variables for profit calculation - use scenario keys directly
    profit = model.addVars(in_sample_scenarios.keys(), name="profit")
    
    # Calculate profit for each scenario
    for s in in_sample_scenarios:
        scenario = in_sample_scenarios[s]
        
        # Initialize profit expression for this scenario
        profit_expr = gp.LinExpr(0)
        
        for h in range(N_HOURS):
            # Wind production, day-ahead price, and balancing price for this scenario and hour
            wind_actual = scenario.loc[h, 'wind']
            price_DA = scenario.loc[h, 'price']
            price_BAL = scenario.loc[h, 'balancing_price']
            
            # Profit calculation for one-price scheme
            # Day-ahead market revenue: price_DA * p_DA[h]
            profit_expr.addTerms(price_DA, p_DA[h])
            
            # Balancing market settlement: price_BAL * (wind_actual - p_DA[h])
            profit_expr.addConstant(price_BAL * wind_actual)  # This is a constant term
            profit_expr.addTerms(-price_BAL, p_DA[h])         # This is a variable term
            
        # Set profit for this scenario
        model.addConstr(profit[s] == profit_expr, f"profit_scenario_{s}")
    
    # Objective: maximize expected profit
    expected_profit = gp.LinExpr()
    for s in in_sample_scenarios:
        expected_profit.addTerms(SCENARIO_PROB, profit[s])
    
    model.setObjective(expected_profit, GRB.MAXIMIZE)
    
    # Solve model
    model.optimize()
    
    # Extract results
    optimal_offers_one_price = [p_DA[h].X for h in range(N_HOURS)]
    scenario_profits = {s: profit[s].X for s in in_sample_scenarios}
    
    return optimal_offers_one_price, model.objVal, scenario_profits